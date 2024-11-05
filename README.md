# Navigating Chaos: An Intelligent Vehicle Anomaly Detection System


## Overview

Navigating Chaos is an innovative project that aims to develop a robust and efficient vehicle anomaly detection system tailored for the unique challenges faced on Indian roads. By leveraging cutting-edge computer vision and machine learning techniques, our system can identify and classify various anomalous behaviors, such as hit-and-run incidents, reckless driving, and footpath riding, in real-time.

## Key Features

1. **Real-time Anomaly Detection**: The system can process video streams in real-time, enabling immediate detection and classification of vehicle anomalies, ensuring timely intervention and prevention of potentially dangerous situations.

2. **Velocity and Trajectory Analysis**: Advanced algorithms for velocity detection and trajectory clustering are implemented to identify reckless driving and other unsafe vehicle behaviors, providing valuable insights for traffic management and safety improvements.

3. **Machine Learning-powered Classification**: Our system employs state-of-the-art machine learning techniques, including Kalman filters, to analyze vehicle trajectories and classify different types of anomalies with high accuracy, enabling a comprehensive and intelligent approach to vehicular behavior monitoring.

4. **Optimized Performance**: By leveraging multi-threading and GPU acceleration, the system is designed to handle the computational demands of real-time processing, ensuring seamless and efficient performance even in high-traffic scenarios.

## Tech Stack

- **Computer Vision**: [YOLO object detection](https://github.com/ultralytics/yolov8), [OpenCV](https://opencv.org/)
- **Machine Learning**: [Kalman Filters](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html), [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- **Parallelization**: [Python Concurrent Futures](https://docs.python.org/3/library/concurrent.futures.html)
- **Development**: Python, NumPy, SciPy


## Roadmap and Future Enhancements

- **Improved Clustering Module**: Explore auto-clustering techniques using silhouette score to enhance the accuracy of anomaly classification.
- **Rule-based Labeling**: Develop a hybrid approach by combining exploratory data analysis and rule-based methods to accurately label each cluster as "rash" or "non-rash".
- **Accident Prediction Model**: Investigate ways to improve the prediction accuracy of accident models, potentially incorporating additional data sources and machine learning algorithms.
- **Hit-and-Run and Footpath Riding Detection**: Complete the implementation of the hit-and-run and footpath riding detection modules to provide a comprehensive solution for identifying various vehicular anomalies.
- **Integration and Deployment**: Seamlessly integrate the individual components into a unified system and prepare the project for real-world deployment, ensuring a smooth and user-friendly experience.

