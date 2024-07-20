  Welcome to the GitHub repository for our research paper on Human Activity Recognition (HAR) using a novel deep learning architecture, WISNet. HAR is a crucial research area with a wide range of innovative applications in healthcare, burglary detection, workplace monitoring, and emergency identification. Traditional approaches often rely on handcrafted features to recognize different human activities, but these methods are limited by their dependency on manually selected features.

  In this repository, we present WISNet, a custom 1D-CNN approach designed to automatically learn relevant features for recognizing six complex human activities: Jogging, Walking Downstairs, Sitting, Standing, Walking, and Climbing Upstairs. Our model incorporates several unique blocks to enhance feature extraction and improve classification performance:

-->Convolved Normalized Pooled (CNPM) Block: Generates significant features from the initial layers.

-->Identity and Basic (IDBN) Block: Extracts residual progressive features to capture complex sequential data dependencies.

-->Channel and Spatial Attention (CASb) Block: Prioritizes or minimizes essential features based on relative weights.

  WISNet achieved an impressive accuracy of 96.41% and an F1-score of 0.95 on the HAR dataset, outperforming existing transfer learning architectures such as GRU, LSTM, and SimpleRNN. We also validated WISNet with similar open-source datasets (UCI-HAR and KU-HAR) and dissimilar open-source datasets (Sleep state detection, Fall detection, and ECG Heartbeat).
