"""Independent assertions on observed prepared examples and diagnostic adapters."""
import numpy as np
import torch
from reassembly.geometry import export_transforms
from reassembly.losses import contact_geometry
from reassembly.training import collate_samples
from . import probes
from .runtime import compare


def sample_contract(sample):
    count=int(sample['fragment_mask'].sum()); anchor=int(sample['anchor_index'])
    xyz=sample['points'][:count].numpy(); raw=sample['original_points'][:count].numpy()
    centers=sample['centroids'][:count].numpy(); scale=float(sample['shared_scale'])
    r=sample['rotations_gt'][:count].numpy(); t=sample['translations_gt'][:count].numpy()
    np.testing.assert_allclose((raw-centers[:,None])/scale,xyz,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(r@r.transpose(0,2,1),np.broadcast_to(np.eye(3),r.shape),atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-6)
    if anchor!=int(np.argmax(np.sqrt((xyz**2).sum(-1).mean(-1)))): raise AssertionError('RMS reference mismatch')
    if not np.array_equal(sample['fragment_mask'].numpy(),np.arange(3)<count): raise AssertionError('Padding membership mismatch')
    labels=sample['fracture_labels'][:count].numpy(); interfaces=sample['interface_ids'][:count].numpy()
    if not np.array_equal(labels>.5,interfaces>=0): raise AssertionError('Fracture/interface labels disagree')
    if count<3:
        assert not sample['points'][count:].any() and not sample['fracture_labels'][count:].any()
        assert torch.all(sample['interface_ids'][count:]==-1)
    exported=export_transforms(r,t,dict(original_points=list(raw),centroids=centers,scale=scale,anchor_index=anchor))
    normalized=np.einsum('fni,fji->fnj',xyz,r)+t[:,None]
    actual=np.stack(exported['aligned_fragments'])
    np.testing.assert_allclose(actual,normalized*scale+centers[anchor],atol=1e-5,rtol=1e-6)
    np.testing.assert_array_equal(exported['transforms'][anchor],np.eye(4))
    discrepancy=np.linalg.norm(normalized-sample['canonical_points'][:count].numpy(),axis=-1)
    if sample['band']!='hard': np.testing.assert_allclose(discrepancy,0,atol=1e-6)
    return {'passed':True,'points':count*len(xyz[0]),'reference':anchor,'shared_scale':scale,
            'gt_to_noise_free_target_distance':probes.distribution(discrepancy),
            'note':'Hard observations may contain recorded noise; GT canonical targets remain noise-free.'}


def matching_contract(model,sample,cfg,device):
    batch=collate_samples([sample],device)
    with torch.no_grad():
        encoded,adapter=probes.predictions(model,batch)
        original=model.match(encoded)
        for a,b in zip(adapter,original):
            for key,value in a.items():
                if isinstance(value,torch.Tensor): torch.testing.assert_close(value,b[key],atol=0,rtol=0)
            if not bool(a['valid'][0]): continue
            mask,distance=contact_geometry(a,batch,cfg['train']['contact_radius'])
            i,j=a['i'],a['j']; ai=a['source_indices'][0].cpu(); bj=a['target_indices'][0].cpu()
            source=sample['canonical_points'][i,ai].numpy(); target=sample['canonical_points'][j,bj].numpy()
            expected_distance=np.linalg.norm(source[:,None]-target[None],axis=-1)
            ids_a=sample['interface_ids'][i,ai].numpy(); ids_b=sample['interface_ids'][j,bj].numpy()
            expected=(ids_a[:,None]>=0)&(ids_a[:,None]==ids_b[None])&(expected_distance<cfg['train']['contact_radius'])
            np.testing.assert_allclose(probes.array(distance)[0],expected_distance,atol=1e-5)
            # Floating-point boundary points cannot establish a label mismatch.
            clear=np.abs(expected_distance-cfg['train']['contact_radius'])>1e-5
            np.testing.assert_array_equal(probes.array(mask)[0][clear],expected[clear])
    return {'passed':True,'unchanged_probability_matrices_exact':True,'targets_independently_checked':True}
