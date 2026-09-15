#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
    std::cout << "=== Monocular Visual Odometry and Mapping ===\n\n";
    // argc contient le nombre d'arguments passés au programme
    // argv contient les arguments eux-mêmes
    // 
    // argv[0] = nom du programme
    // argv[1] = première image
    // argv[2] = deuxième image
    if (argc < 3)
    {
        std::cerr << "Usage: monocular_visual_odometry "
                  << "<image_1> <image_2>\n";
        return 1;
    }
    
    // Chemins donnés par l'utilisateur
    const std::string imagePath1 = argv[1];
    const std::string imagePath2 = argv[2];

    // Chargement en niveau de gris
    //
    // Une image en bdg ne contient qu'un canal d'intensité, contrairement à une image couleur
    // contenant généralement 3 canaux RGB dans OpenCV.
    cv::Mat image1 = cv::imread(
        imagePath1,
        cv::IMREAD_GRAYSCALE
    );

    cv::Mat image2 = cv::imread(
        imagePath2,
        cv::IMREAD_GRAYSCALE
    );


    // SI OpenCV n'a pas réussi à charger les images, cv::Mat sera vide.
    if (image1.empty() || image2.empty())
    {
        std::cerr << "Error: could not load one or both images.\n";
        return 1;
    }

    std::cout << "Images loaded successfully.\n";

    // image.cols = largeur
    // image.rows = hauteur
    std::cout << "Image 1: "
              << image1.cols << " x " << image1.rows << '\n';

    std::cout << "Image 2: "
              << image2.cols << " x " << image2.rows << '\n';


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
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;


    // Cette matrice contiendra les descripteurs ORB.
    //
    // Chaque ligne correspondra à un keypoint.
    // Pour ORB, un descripteur fait 32 octets = 256 bits.
    cv::Mat descriptors1;
    cv::Mat descriptors2;


    // detectAndCompute réalise 2 opérations :
    //
    // 1. détecter les points d'intérêt
    // 2. calculer un descripteur autour de chaque point
    // 
    // cv::noArray() signifie qu'on n'utilise aucun masque :
    // toute l'image peut être analysée.
    orb->detectAndCompute(
        image1,
        cv::noArray(),
        keypoints1,
        descriptors1
    );


    orb->detectAndCompute(
        image2,
        cv::noArray(),
        keypoints2,
        descriptors2
    );


    std::cout << "Image 1 keypoints: " 
              << keypoints1.size()
              << '\n';

    std::cout << "Image 2 keypoints: "
              << keypoints2.size()
              << '\n';


    // -------------------------------------------------------
    // Création du matcher
    // 
    // BFMatcher = Brute-Force Matcher
    // 
    // Pour chaque descripteur de l'image 1, il compare les
    // descripteurs de l'image 2 afin de trouver le plus proche
    // 
    // ORB fournit des descripteurs binaires.
    // Donc on utilisera la distance de Hamming.
    // 
    // true = crossCheck;
    // A doit choisir B ET B doit choisir A.
    // Cela élimine déjà beaucoup de mauvaises correspondances.
    // --------------------------------------------------------
    
    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        true
    );

    std::vector<cv::DMatch> matches;

    matcher.match(
        descriptors1,
        descriptors2,
        matches
    );

    std::cout << "Raw matches: "
              << matches.size()
              << '\n';

    // ---------------------------------------------------------
    // Tri des correspondances
    // 
    //  Une petite distance = deux descripteurs très similaires.
    // 
    // On place donc les meilleurs matches en premier.
    // ---------------------------------------------------------

    std::sort(
        matches.begin(),
        matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b)
        {
            return a.distance < b.distance;
        }
    );


    // ---------------------------------------------------------
    // Conservation des meilleurs matches pour l'affichage
    // 
    // On ne jette pas encore les autres de manière définitive.
    // On veut juste une image lisible.
    // ---------------------------------------------------------

    const std::size_t numberOfMatchesToDisplay = 
        std::min<std::size_t>(100, matches.size());

    std::vector<cv::DMatch> bestMatches(
        matches.begin(),
        matches.begin() + numberOfMatchesToDisplay
    );

    std::cout << "Matches displayed: "
              << bestMatches.size()
              << '\n';

    // -------------------------------------------------------
    // Visualisation
    // 
    // drawMatches place les deux images côte à côte et relie
    // chaque paire de points correspondants par une ligne.
    // ------------------------------------------------------- 
    cv::Mat visualization;

    cv::drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        bestMatches,
        visualization
    );


    // Réduction de la visualisation pour l'adapter à mon PC
    const int displayWidth = 1100;

    // Facteur d'échelle pour mon PC
    const double scale =
        static_cast<double>(displayWidth) / visualization.cols;

    // Conservation du rapport largeur/hauteur
    const int displayHeight = 
    static_cast<int>(visualization.rows * scale);

    cv::Mat resizedVisualization;

    cv::resize(
        visualization,
        resizedVisualization,
        cv::Size(displayWidth, displayHeight)
    );

    cv::imshow("ORB feature matches", resizedVisualization);

    // Attend qu'une touche soit pressée
    cv::waitKey(0);

    return 0;

}