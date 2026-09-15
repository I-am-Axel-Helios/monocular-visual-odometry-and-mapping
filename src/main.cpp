#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
    std::cout << "=== Monocular Visual Odometry and Mapping ===\n\n";
    // argc contient le nombre d'arguments passés au programme
    // argv contient les arguments eux-mêmes
    // 
    // argv[0] = nom du programme
    // argv[1] = chemin de l'image fourni dans le terminal
    if (argc < 2)
    {
        std::cerr << "Usage: monocular_visual_odometry <image_path>\n";
        return 1;
    }
    
    // Chemin donné par l'utilisateur
    const std::string imagePath = argv[1];

    // Chargement en niveau de gris
    //
    // Une image en bdg ne contient qu'un canal d'intensité, contrairement à une image couleur
    // contenant généralement 3 canaux RGB dans OpenCV.
    cv::Mat image = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );


    // SI OpenCV n'a pas réussi à charger l'image, cv::Mat sera vide.
    if (image.empty())
    {
        std::cerr << "Error: could not load image: "
                  << imagePath << '\n';

        return 1;
    }

    std::cout << "Image loaded successfully.\n";

    // image.cols = largeur
    // image.rows = hauteur
    std::cout << "Resolution: "
              << image.cols << " x "
              << image.rows << '\n';

    // Affiche l'image dans une fenêtre OpenCV
    cv::imshow("Input image", image);

    // Attend qu'une touche soit pressée
    cv::waitKey(0);

    return 0;

}