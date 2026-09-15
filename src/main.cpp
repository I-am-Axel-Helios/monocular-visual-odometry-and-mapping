#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <vector>


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


    // ------------------------------------------
    // Détection de points d'intérêt avec ORB
    // ------------------------------------------

    // On demande à ORB de conserver au max 1000 points
    // 
    // cv::Ptr est un pointeur intelligent de OpenCV
    // Il gère automatiquement la durée de vie de l'objet ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);


    // Un Keypoint représente un point d'intérêt détecté dans l'image.
    //
    // Il contient notamment :
    //  - sa position (x,y)
    //  - sa taille
    //  - son orientation
    //  - une mesure de sa qualité ("response")
    std::vector<cv::KeyPoint> keypoints;


    // Cette matrice contiendra les descripteurs ORB.
    //
    // Chaque ligne correspondra à un keypoint.
    // Pour ORB, un descripteur fait 32 octets = 256 bits.
    cv::Mat descriptors;


    // detectAndCompute réalise 2 opérations :
    //
    // 1. détecter les points d'intérêt
    // 2. calculer un descripteur autour de chaque point
    // 
    // cv::noArray() signifie qu'on n'utilise aucun masque :
    // toute l'image peut être analysée.
    orb->detectAndCompute(
        image,
        cv::noArray(),
        keypoints,
        descriptors
    );


    std::cout << "ORB keypoints detected: " 
              << descriptors.rows
              << " x "
              << descriptors.cols
              << '\n';

    // Image qui contiendra l'image originale et les points ORB dessinés
    cv::Mat visualization;


    // Dessine les keypoints détectés
    // 
    // DRAW_RICH_KEYPOINTS permet notamment de visualiser leur taaille et leur orientation
    cv::drawKeypoints(
        image,
        keypoints,
        visualization,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cv::imshow("ORB keypoints", visualization);

    // Attend qu'une touche soit pressée
    cv::waitKey(0);

    return 0;

}