Coronary artery disease (CAD) is an exceptionally prevalent condition with high mortality rates. It is the
leading cause of death in both the United States and around the world. Pulmonary computed tomography
(CT) scans and invasive coronary angiograms are accredited as the main methods of CAD diagnosis. However,
there are many downsides, such as cost and exposure to ionizing radiation, associated with these methods.
Thanks to technological advancements, machine learning, particularly neural networks, has gained prominence
in medical image analysis due to its precision and accuracy in diagnosing various ailments. This study provides
a novel approach by implementing convolutional neural networks (CNNs) to determine if patients have CAD
via binary classification. We initiated an InceptionV3 pre-trained model, named CADD, paired with custom
layers to correctly and effectively diagnose CAD. Data preprocessing is essential to our model’s precision. Mosaic
projection view (MPV) is the primary input format because of its relevance and interoperability, which can be
leveraged by our model. Numerous augmentations were applied to the dataset and the model was fine-tuned with
multiple iterations, resulting in a significant increase in accuracy. The dataset used to train CADD consisted
of thousands of MPVs and is publicly accessible. Our training and testing data split was approximately 80-20.
CADD achieved a final accuracy of 94.97% compared to our base model’s 88.26% on the same dataset.
