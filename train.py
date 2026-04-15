import os, time, glob, torch, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split

from preprocessor import CSIPreprocessor
from dataset import WiFiPoseDataset, pose_collate_fn
from model import WiFiEnd2EndPoseNet
from loss import End2EndPoseLoss
from matcher import hungarian_matching_pose

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = {'total': 0, 'count': 0, 'heatmap': 0, 'conf': 0}
    for batch in loader:
        preds = model(batch['csi'].to(device))
        tgts = {'count': batch['count'].to(device), 'heatmaps': batch['heatmaps'].to(device), 'mask': batch['mask'].to(device)}
        loss, ldict = criterion(preds, tgts)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        for k in losses: losses[k] += ldict[k]
    return {k: v / len(loader) for k, v in losses.items()}

def validate_epoch(model, loader, criterion, device):
    model.eval()
    losses = {'total': 0, 'count': 0, 'heatmap': 0, 'conf': 0}
    correct_cnt, total_samples, pred_hist = 0, 0, {0:0, 1:0, 2:0, 3:0}
    total_pck, total_kpts, RS_IDX, LH_IDX, THRESH = 0.0, 0, 6, 11, 0.2

    with torch.no_grad():
        for batch in loader:
            csi, cnt_gt, hm_gt, mask = (batch['csi'].to(device), batch['count'].to(device),
                                        batch['heatmaps'].to(device), batch['mask'].to(device))
            preds = model(csi)
            tgts = {'count': cnt_gt, 'heatmaps': hm_gt, 'mask': mask}
            loss, ldict = criterion(preds, tgts)
            for k in losses: losses[k] += ldict[k]

            B = csi.size(0)
            pred_cnt = torch.argmax(preds['count_logits'], dim=1)
            correct_cnt += (pred_cnt == cnt_gt).sum().item(); total_samples += B
            for p in pred_cnt.cpu().numpy(): pred_hist[int(p)] += 1

            if mask.sum() > 0:
                match_idx, match_valid = hungarian_matching_pose(preds['pred_heatmaps'], hm_gt, mask)
            else: continue

            for b in range(B):
                if mask[b].sum() == 0: continue
                for p_idx in range(model.max_persons):
                    if not match_valid[b, p_idx]: continue
                    gt_idx = match_idx[b, p_idx]
                    rs_y, rs_x = torch.unravel_index(hm_gt[b, gt_idx, RS_IDX].argmax(), hm_gt.shape[3:])
                    lh_y, lh_x = torch.unravel_index(hm_gt[b, gt_idx, LH_IDX].argmax(), hm_gt.shape[3:])
                    torso_len = torch.norm(torch.tensor([rs_x.float(), rs_y.float()]) - torch.tensor([lh_x.float(), lh_y.float()]))
                    if torso_len < 2.0: continue

                    for k in range(model.num_keypoints):
                        dist = torch.norm(torch.tensor(torch.unravel_index(preds['pred_heatmaps'][b, p_idx, k].argmax(), preds['pred_heatmaps'].shape[3:])).float() -
                                          torch.tensor(torch.unravel_index(hm_gt[b, gt_idx, k].argmax(), hm_gt.shape[3:])).float())
                        if dist / torso_len <= THRESH: total_pck += 1
                        total_kpts += 1

    return {k: v / len(loader) for k, v in losses.items()}, correct_cnt / max(1, total_samples), total_pck / max(1, total_kpts), pred_hist

def main():
    CFG = dict(DATA_ROOT='/mnt/X/czq/CV/姿态估计（HPE）/多人姿态估计/数据集（1-3）人/MHPE_dataset',
               IMG_SIZE=(1920,1080), HM_SIZE=(56,56), MAX_P=3, KPTS=17, BATCH=32, EPOCHS=100, LR=3e-4,
               VAL_SPLIT=0.2, NEG_RATIO=0.0, USE_HUN=True, SEED=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pre = CSIPreprocessor()
    samples = []
    for sub in ['one','two','three']:
        sp = os.path.join(CFG['DATA_ROOT'], sub)
        if not os.path.exists(sp): continue
        for sess in sorted(glob.glob(os.path.join(sp, '*/')))[:1]:
            md = os.path.join(sess, 'mags')
            if os.path.exists(md):
                for f in sorted([x for x in os.listdir(md) if x.endswith('.npy')])[:100]:
                    samples.append(np.load(os.path.join(md, f)))
    pre.fit(samples)

    tr_subs, va_subs = [], []
    for sub in ['one','two','three']:
        sr = os.path.join(CFG['DATA_ROOT'], sub)
        if not os.path.exists(sr): continue
        ds_list = [WiFiPoseDataset(s, pre, CFG['IMG_SIZE'], CFG['HM_SIZE'], CFG['MAX_P'], CFG['KPTS'], CFG['NEG_RATIO']) for s in sorted(glob.glob(os.path.join(sr, '*/')))]
        if not ds_list: continue
        full = ConcatDataset(ds_list)
        gen = torch.Generator().manual_seed(CFG['SEED'])
        vl = int(len(full) * CFG['VAL_SPLIT'])
        tr, va = random_split(full, [len(full)-vl, vl], generator=gen)
        tr_subs.append(tr); va_subs.append(va)

    tr_dl = DataLoader(ConcatDataset(tr_subs), CFG['BATCH'], shuffle=True, num_workers=8, collate_fn=pose_collate_fn, pin_memory=True)
    va_dl = DataLoader(ConcatDataset(va_subs), CFG['BATCH'], shuffle=False, num_workers=8, collate_fn=pose_collate_fn, pin_memory=True)

    model = WiFiEnd2EndPoseNet(CFG['MAX_P'], CFG['KPTS'], 512, CFG['HM_SIZE']).to(device)
    criterion = End2EndPoseLoss(alpha_heatmap=100.0, use_hungarian=CFG['USE_HUN'], heatmap_mode="peak_mse")
    opt = optim.AdamW(model.parameters(), lr=CFG['LR'], weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    best_pck, best_ep = 0.0, 0

    print("\n--- Start end-to-end training ---")
    for ep in range(CFG['EPOCHS']):
        t0 = time.time()
        tl = train_epoch(model, tr_dl, criterion, opt, device)
        vl, acc, pck, ph = validate_epoch(model, va_dl, criterion, device)
        sched.step()
        print(f"Epoch {ep+1:3d}/{CFG['EPOCHS']} | T:{tl['total']:.4f} V:{vl['total']:.4f} | "
              f"lc:{vl['count']:.3f} lh:{vl['heatmap']:.3f} lf:{vl['conf']:.3f} | Acc:{acc:.1%} PCK:{pck:.3f} | "
              f"LR:{opt.param_groups[0]['lr']:.2e} T:{time.time()-t0:.1f}s")
        if pck > best_pck:
            best_pck, best_ep = pck, ep+1
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                        'val_loss': vl['total'], 'accuracy': acc, 'pck': pck, 'preprocessor': {'mean': pre.mean, 'std': pre.std}},
                       'best_model.pth')
            print(f"  ✓ [Saved] Best model (PCK:{pck:.3f})")
    print(f"\nTraining completed! Best PCK: {best_pck:.3f} (Epoch {best_ep})")

if __name__ == '__main__':
    main()