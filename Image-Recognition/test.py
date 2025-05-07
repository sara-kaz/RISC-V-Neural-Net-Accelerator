# test.py

from train import PipelineCPU
import numpy as np
import matplotlib.pyplot as plt

def visualize_mnist_image(image):
    """Visualize a MNIST image"""
    plt.figure(figsize=(3, 3))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    # Create CPU instance
    cpu = PipelineCPU()

    # Load MNIST data
    X_train, X_test, y_train, y_test = cpu.load_mnist_data()

    # Initialize weights
    cpu.initialize_weights()

    # Test on a few images
    num_test_images = 10
    for i in range(num_test_images):
        test_image = X_test[i]
        actual_label = y_test[i]

        # Visualize the image
        print(f"\nTesting image {i+1}/{num_test_images}")
        print(f"Actual digit: {actual_label}")
        visualize_mnist_image(test_image)

        # Make prediction
        print("Making prediction...")
        prediction = cpu.predict(test_image)
        print(f"Predicted digit: {prediction}")

        # Check if prediction is correct
        if prediction == int(actual_label):
            print("✅ Correct prediction!")
        else:
            print("❌ Incorrect prediction!")

        # Wait for user input before proceeding to next image
        input("\nPress Enter to continue to next image...")

if __name__ == "__main__":
    main()
