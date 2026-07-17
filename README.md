# FlexGait
FlexGait: Fusion-guided Unimodal Learning for Flexible Diagnosis of Anterior Cruciate Ligament Deficiency

<!-- https://www.xxxxxxxxx.com/

--------------------

Citation:
```
Authors et al. OmniECG: Single-source pretraining on electrocardiograms generalizes to heterogeneous medical time series
journal. doi
```

Bibtex:
```
@article{,
  title = {ECG-Derived Time Series Foundation Model: A Domain Adaptation Paradigm for Medical Signal Analysis},
  author = {},
  year = {},
  volume = {},
  pages = {},
  doi = {},
  journal = {},
  number = {}
}
```
-------------------- -->

## Abstract information
Background: Anterior cruciate ligament deficiency (ACLD) can lead to persistent alterations in lower-limb neuromuscular control during walking. Although gait signals and video-derived skeletal motion provide complementary information, conventional multimodal models often require all modalities to be available during inference, limiting their use in real-world clinical settings with incomplete data.

Methods: We developed FlexGAIT, a fusion-guided multimodal learning framework for gait-based ACLD identification. FlexGAIT integrates clinically acquired bilateral six-degree-of-freedom gait signals and video-derived lower-limb skeletal sequences while preserving modality-specific branches for signal-only and video-only inference. The model was trained and evaluated using participant-level data splits from an internal cohort and independently validated on an external cohort. The development cohort included 273 training participants, 86 validation participants and 91 internal test participants, and the external cohort included 195 participants. FlexGAIT was assessed under multimodal, signal-only and video-only inference settings and compared with independently trained unimodal deep-learning baselines.

Results: In the internal test cohort, the fusion pathway achieved an accuracy of 0.967, ROC-AUC of 0.996 and PR-AUC of 0.995. In the external cohort, it maintained robust performance, with an accuracy of 0.913, ROC-AUC of 0.974 and PR-AUC of 0.975. The signal-only branch achieved ROC-AUCs of 0.991 and 0.943 on the internal and external cohorts, respectively, whereas the video-only branch achieved ROC-AUCs of 0.893 and 0.932. Both modality-specific FlexGAIT branches outperformed independently trained unimodal models. Ablation, representation and decision-curve analyses further supported the contribution of cross-modal learning and the potential clinical utility of the framework.

Conclusions: FlexGAIT enables flexible gait-based ACLD identification across multimodal and unimodal scenarios. By using multimodal learning to strengthen modality-specific inference, FlexGAIT may provide an objective computational approach for ACLD assessment under variable clinical data-availability conditions.

## Key words
Anterior cruciate ligament deficiency; Gait analysis; Multimodal learning; Flexible inference; Missing modality

## Requirements

This code was tested on Python 3 with tensorflow

In addition, the packages we are calling now is as follows:
- [x] tensorflow
- [x] sklearn
- [x] random
- [x] scipy
- [x] pandas
- [x] numpy

## Framework illustration
Gait assessment provides a non-invasive window into functional impairment after anterior cruciate ligament deficiency, but existing learning-based approaches are often constrained by fixed input requirements and limited ability to operate when one modality is unavailable. Here, we developed FlexGAIT, a fusion-guided multimodal learning framework that integrates clinically acquired gait signals with video-derived skeletal sequences while preserving modality-specific inference pathways. Our results show that multimodal gait learning can be used not only to improve ACLD identification when complete data are available, but also to strengthen single-modality inference under realistic incomplete-input scenarios. This shifts multimodal gait analysis from a rigid deployment setting requiring complete inputs towards a more flexible framework that can adapt to different clinical data-availability conditions.
![Framework Illustration](./figure/Figure_all_new1_01(1)_2477.png)
