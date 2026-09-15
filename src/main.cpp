#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
    std::cout << "\n\n=== Monocular Visual Odometry and Mapping ===\n\n";
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


    // ------------------------------------------------------------
    // Matching KNN des descripteurs ORB
    // 
    // Au lieu de demander uniquement le MEILLEUR candidat,
    // on va demander les 2 meilleurs candidats pour chaque 
    // descripteur de l'image 1.
    // 
    // crossCheck doit être desactivé pour utiliser cette stratégie.
    // ------------------------------------------------------------
    
    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    // Pour chaque descripteur de l'image 1,
    // nous allons recevoir jusqu'à 2 candidats dans l'image 2.
    std::vector<std::vector<cv::DMatch>> knnMatches;

    matcher.knnMatch(
        descriptors1,
        descriptors2,
        knnMatches,
        2
    );

    std::cout << "KNN match groups: "
              << knnMatches.size()
              << '\n';


    // ---------------------------------------------------------
    // Lowe ratio test
    // ---------------------------------------------------------

    const float ratioThreshold = 0.75f;

    std::vector<cv::DMatch> goodMatches;

    for (const auto& candidates : knnMatches)
    {
        // Il faut au moins 2 candidats pour calculer le ratio.
        if (candidates.size() < 2)
        {
            continue;
        }

        const cv::DMatch& bestMatch = candidates[0];
        const cv::DMatch& secondMatch = candidates[1];

        // On garde le meilleur candidat uniquement s'il est 
        // nettement meilleur que le deuxieme
        if (bestMatch.distance < ratioThreshold * secondMatch.distance)
        {
            goodMatches.push_back(bestMatch);
        }
    }

    std::cout << "Matches after ratio test: "
              << goodMatches.size()
              << '\n';


    // ---------------------------------------------------------
    // Tri des correspondances
    // 
    //  Une petite distance = deux descripteurs très similaires.
    // 
    // On place donc les meilleurs matches en premier.
    // ---------------------------------------------------------

    std::sort(
        goodMatches.begin(),
        goodMatches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b)
        {
            return a.distance < b.distance;
        }
    );


    // ----------------------------------------------------------------
    // Préparation des correspondances géométriques
    //
    // findFundamentalMat ne travaille pas directement avec DMatch
    // Il lui faut les coordonnées 2D des points correspondants
    // ----------------------------------------------------------------

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    points1.reserve(goodMatches.size());
    points2.reserve(goodMatches.size());

    for (const cv::DMatch& match : goodMatches)
    {
        // queryIdx -> point dans l'image 1
        // trainIdx -> point correspondant dans l'image 2
        points1.push_back(
            keypoints1[match.queryIdx].pt
        );

        points2.push_back(
            keypoints2[match.trainIdx].pt
        );
    }


    // ---------------------------------------------------------
    // Estimation robuste de la matrice fondamentale avec RANSAC
    // 
    // fundamentalMask contiendra pour chaque correspondance :
    //  1 -> inlier, compatible avec la géometrie estimée
    //  0 -> outlier, rejetée
    // ---------------------------------------------------------

    cv::Mat fundamentalMask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(
        points1,
        points2,

        // RANSAC permet d'estimer F malgré la présence d'outliers.
        cv::FM_RANSAC,

        // Distance minimale, en pixels, pour considérer qu'un point
        // respecte suffisamment la contraite épipolaire.
        1.5,

        // Niveau de confiance demandé à RANSAC.
        0.99,

        fundamentalMask
    );


    // ---------------------------------------------------------
    // Conservation uniquement des matches validés par RANSAC
    // ---------------------------------------------------------

    std::vector<cv::DMatch> inliersMatches;

    for (std::size_t i = 0; i < goodMatches.size(); ++i)
    {
        if (fundamentalMask.at<uchar>(static_cast<int>(i)) != 0)
        {
            inliersMatches.push_back(goodMatches[i]);
        }
    }

    std::cout << "Matches before RANSAC: "
              << goodMatches.size()
              << '\n';

    std::cout << "Geometric inliers: "
              << inliersMatches.size()
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
        inliersMatches,
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