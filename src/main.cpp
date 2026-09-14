#include <iostream>

// OpenCV
#include <opencv2/opencv.hpp>

// Eigen : bibliothèque d'algèbre linéaire
#include <Eigen/Dense>

int main()
{
    std::cout << "=== Monocular Visual Odometry and Mapping ===\n\n";

    // ------------------------------------------------------------
    // Test 1 : vérifier que notre programme utilise bien OpenCV.
    // CV_VERSION est une constante fournie par OpenCV.
    // ------------------------------------------------------------
    std::cout << "OpenCV version : "
              << CV_VERSION
              << '\n';

    // ------------------------------------------------------------
    // Test 2 : vérifier qu'Eigen fonctionne.
    //
    // Matrix3d signifie :
    //      Matrix -> matrice
    //      3      -> 3 lignes / 3 colonnes
    //      d      -> valeurs de type double
    //
    // Ici nous construisons une matrice 3x3 comparable à une
    // matrice intrinsèque de caméra K.
    // ------------------------------------------------------------
    Eigen::Matrix3d K;

    K << 718.856, 0.0,     607.1928,
         0.0,     718.856, 185.2157,
         0.0,     0.0,     1.0;

    std::cout << "\nCamera matrix K:\n";
    std::cout << K << '\n';

    // ------------------------------------------------------------
    // Calcul simple avec Eigen :
    // déterminant de la matrice.
    // Le but est seulement de vérifier que les opérations
    // mathématiques d'Eigen sont réellement disponibles.
    // ------------------------------------------------------------
    std::cout << "\ndet(K) = "
              << K.determinant()
              << '\n';

    return 0;
}