#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
    std::cout << "\n\n=== Monocular Visual Odometry and Mapping ===\n\n";
    // --------------------------------------------------------
    // Arguments du programme
    // 
    // argv[0] = nom du programme
    // argv[1] = première image
    // argv[2] = deuxième image
    // argv[3] = fichier de calibration caméra
    // --------------------------------------------------------
    if (argc < 4)
    {
        std::cerr << "Usage: monocular_visual_odometry "
                  << "<image_1> <image_2> <calibration_file>\n";
        return 1;
    }
    
    // Chemins donnés par l'utilisateur
    const std::string imagePath1 = argv[1];
    const std::string imagePath2 = argv[2];
    const std::string calibrationPath = argv[3];


    // --------------------------------------------------------
    // Chargement des images en niveau de gris
    //
    // Une image en bdg ne contient qu'un canal d'intensité, 
    // contrairement à une image couleur contenant généralement 
    // 3 canaux BGR dans OpenCV.
    // --------------------------------------------------------
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
              << image1.cols 
              << " x " 
              << image1.rows 
              << '\n';

    std::cout << "Image 2: "
              << image2.cols 
              << " x " 
              << image2.rows 
              << '\n';

    // --------------------------------------------------------
    // Ouverture du fichier de calibration
    // --------------------------------------------------------
    cv::FileStorage calibrationFile(
        calibrationPath,
        cv::FileStorage::READ
    );

    // En cas de problème lors du chargement de la calibration
    if (!calibrationFile.isOpened())
    {
        std::cerr << "Error: could not open calibration file: "
            << calibrationPath
            << '\n';
        return 1;
    }


    // --------------------------------------------------------
    // Lecture des paramètres intrinsèques
    // --------------------------------------------------------
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;

    int calibrationWidth = 0;
    int calibrationHeight = 0;

    calibrationFile["camera_matrix"]
        >> cameraMatrix;

    calibrationFile["distortion_coefficients"]
        >> distortionCoefficients;

    calibrationFile["image_width"]
        >> calibrationWidth;

    calibrationFile["image_height"]
        >> calibrationHeight;

    calibrationFile.release();

    
    // --------------------------------------------------
    // Vérification de la calibration
    // --------------------------------------------------
    if (cameraMatrix.empty() || 
        cameraMatrix.rows != 3 ||
        cameraMatrix.cols != 3 || 
        distortionCoefficients.empty())
    {
        std::cerr << "Error: invalid calibration data.\n";
        return 1;
    }

    // ---------------------------------------------------
    // Vérification de la résolution
    // ---------------------------------------------------
    if (image1.cols != calibrationWidth ||
        image1.rows != calibrationHeight ||
        image2.cols != calibrationWidth ||
        image2.rows != calibrationHeight)
    {
        std::cerr << "Calibration: "
                  << calibrationWidth
                  << " x "
                  << calibrationHeight
                  << '\n';

        std::cerr << "Image1: "
                  << image1.cols
                  << " x "
                  << image1.rows
                  << '\n';

        std::cerr << "Image 2: "
                  << image2.cols
                  << " x "
                  << image2.rows
                  << '\n';
        return 1;
    }


    // ---------------------------------------------------
    // Affichage de la calibration utilisée
    // ---------------------------------------------------
    std::cout << "\nCamera matrix K:\n"
              << cameraMatrix
              << "\n\n";

    std::cout << "Distortion coefficients:\n"
              << distortionCoefficients
              << "\n\n";
    
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
    // Une petite distance = deux descripteurs très similaires.
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

    if (points1.size() < 8)
    {
        std::cerr << "Error: not enough correspondences "
                  << "to estimate the fundamental matrix.\n";
        return 1;
    }

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

    if (fundamentalMatrix.empty()) 
    {
        std::cerr << "Error: fundamental matrix estimation failed.\n";
        return 1;
    }


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

    // --------------------------------------------------------
    // Préparation des correspondances pour l'estimation de la 
    // pose de la caméra
    // 
    // On ne conserve ici que les correspondances déjà validées
    // par l'estimation robuste de la matrice fondamentale
    // --------------------------------------------------------
    std::vector<cv::Point2f> posePoints1;
    std::vector<cv::Point2f> posePoints2;

    posePoints1.reserve(inliersMatches.size());
    posePoints2.reserve(inliersMatches.size());

    for (const cv::DMatch& match : inliersMatches)
    {
        posePoints1.push_back(
            keypoints1[match.queryIdx].pt
        );

        posePoints2.push_back(
            keypoints2[match.trainIdx].pt
        );
    }


    // --------------------------------------------------------
    // Correction de la distorsion de la caméra
    //
    // Les coordonnées des keypoints viennent directement des
    // images réelles et sont donc affectées par la distortions.
    // 
    // undistortPoints() calcule leurs positions corrigées.
    // --------------------------------------------------------
    std::vector<cv::Point2f> undistortedPoints1;
    std::vector<cv::Point2f> undistortedPoints2;

    cv::undistortPoints(
        posePoints1,
        undistortedPoints1,
        cameraMatrix,
        distortionCoefficients,
        cv::noArray(),
        cameraMatrix
    );

    cv::undistortPoints(
        posePoints2,
        undistortedPoints2,
        cameraMatrix,
        distortionCoefficients,
        cv::noArray(),
        cameraMatrix
    );


    // --------------------------------------------------------
    // Estimation de la matrice essentielle E
    // 
    // Pour une caméra calibrée : E = [t]x R
    // 
    // Elle encode la rotation et la direction de translation
    // entre les deux positions de la caméra.
    // --------------------------------------------------------
    cv::Mat essentialMask;

    cv::Mat essentialMatrix = cv::findEssentialMat(
        undistortedPoints1,
        undistortedPoints2,
        cameraMatrix,
        cv::RANSAC,

        // Proba demandée à RANSAC de trouver un modèle valide
        0.999,

        // Seuil géométrique en pixels
        1.0,

        essentialMask
    );


    if (essentialMatrix.empty())
    {
        std::cerr << "Error: essential matrix estimation failed.\n";
        return 1;
    }

    std::cout << "\nEssential matrix E:\n"
              << essentialMatrix
              << '\n';

    // --------------------------------------------------------
    // Récupération de la pose relative
    // 
    // recoverPose() décompose E pour obtenir:
    // R : rotation relative entre les deux caméras
    // t : direction de translation
    // 
    // Attention: l norme réelle de la translation est inconnue
    // en vision monoculaire.
    // --------------------------------------------------------
    cv::Mat rotation;
    cv::Mat translation;

    // On copie le masque car recoverPose() peut le modifier.
    // Après cet appel, poseMask représentera les points qui sont
    // à la fois compatibles avec E et avec une reconstruction
    // physiquement valide devant les deux caméras.
    cv::Mat poseMask = essentialMask.clone();

    const int poseInliers = cv::recoverPose(
        essentialMatrix,
        undistortedPoints1,
        undistortedPoints2,
        cameraMatrix,
        rotation,
        translation,
        poseMask
    );

    std::cout << "\n=== Relative camera pose ===\n\n";

    std::cout << "Rotation R:\n"
              << rotation
              << "\n\n";

    std::cout << "Translation direction t:\n"
              << translation
              << "\n\n";

    std::cout << "Pose inliers: "
              << poseInliers
              << " / "
              << undistortedPoints1.size()
              << '\n';

    std::cout << "det(R): "
              << cv::determinant(rotation)          // Valeur attendue: 1 (pour une vraie rotation))
              << '\n';

    std::cout << "||t||: "
              << cv::norm(translation)
              << '\n';

    // --------------------------------------------------------
    // Conservation des matches validés par recoverPose()
    // 
    // Ce seront les meilleurs candidats pour notre
    // triangulation 3D.
    // --------------------------------------------------------
    std::vector<cv::DMatch> poseInliersMatches;

    for (std::size_t i = 0; i < inliersMatches.size(); ++i)
    {
        if (poseMask.at<uchar>(static_cast<int>(i)) != 0)
        {
            poseInliersMatches.push_back(
                inliersMatches[i]
            );
        }
    }

    std::cout << "Final pose inlier matches: "
              << poseInliersMatches.size()
              << '\n';

    // -------------------------------------------------------
    // Préparation des points pour la triangulation
    // 
    // poseMask correspond aux points utilisés dans les 
    // vecteurs undistortedPoints.
    // 
    // On ne conserve que ceux validés par recoverPose().
    // -------------------------------------------------------
    std::vector<cv::Point2f> triangulationPoints1;
    std::vector<cv::Point2f> triangulationPoints2;

    triangulationPoints1.reserve(poseInliers);
    triangulationPoints2.reserve(poseInliers);

    for (std::size_t i = 0; i < undistortedPoints1.size(); ++i)
    {
        if (poseMask.at<uchar>(static_cast<int>(i)) != 0)
        {
            triangulationPoints1.push_back(
                undistortedPoints1[i]
            );

            triangulationPoints2.push_back(
                undistortedPoints2[i]
            );
        }
    }

    std::cout << "Points prepared for triangulation: "
              << triangulationPoints1.size()
              << '\n';

    // Poses des deux photos
    // Première caméra: P1 = K [ I | 0 ]
    cv::Mat firstCameraPose = 
        cv::Mat::zeros(
            3,
            4,
            CV_64F
        );

    cv::Mat identity = 
        cv::Mat::eye(
            3,
            3,
            CV_64F
        );

    identity.copyTo(
        firstCameraPose(
            cv::Rect(0, 0, 3, 3)
        )
    );

    cv::Mat projectionMatrix1 = cameraMatrix * firstCameraPose;

    
    // Seconde caméra : P2 = K [ R | t ]
    cv::Mat secondCameraPose;

    cv::hconcat(
        rotation,
        translation,
        secondCameraPose
    );

    cv::Mat projectionMatrix2 = cameraMatrix * secondCameraPose;

    std::cout << "\nProjection matrix P1:\n"
              << projectionMatrix1
              << "\n\n";

    std::cout << "\nProjection matrix P2:\n"
              << projectionMatrix2
              << "\n\n";

    // ----------------------------------------------------------
    // Triangulation
    // 
    // OpenCV reconstruit les points 3D en coordonnées homogènes
    // à partir des 2 matrices de projection et des observations 
    // 2D correspondantes. 
    // ----------------------------------------------------------
    cv::Mat homogeneousPoints4D;

    cv::triangulatePoints(
        projectionMatrix1,
        projectionMatrix2,
        triangulationPoints1,
        triangulationPoints2,
        homogeneousPoints4D
    );


    cv::Mat homogeneousPoints4D64;

    homogeneousPoints4D.convertTo(
        homogeneousPoints4D64,
        CV_64F
    );


    // --------------------------------------------------------
    // Passage des coordonnées homogènes aux coordonnées 3D
    // euclidiennes.
    // 
    // (Xh, Yh, Zh, Wh) -> (Xh/Wh, Yh/Wh, Zh/Wh)
    // --------------------------------------------------------
    std::vector<cv::Point3d> triangulatedPoints;

    // Observations 2D correspondant exactement aux points 3D
    // qui ont survécu à tous nos filtres
    std::vector<cv::Point2f> validTriangulationPoints1;
    std::vector<cv::Point2f> validTriangulationPoints2;

    triangulatedPoints.reserve(
        homogeneousPoints4D64.cols
    );

    validTriangulationPoints1.reserve(
        homogeneousPoints4D64.cols
    ); 
    
    validTriangulationPoints2.reserve(
        homogeneousPoints4D64.cols
    );


    for (int i = 0; i < homogeneousPoints4D64.cols; ++i)
    {
        const double w = homogeneousPoints4D64.at<double>(3, i);

        // Au cas où w est quasi-nul, la division est instable.
        if (std::abs(w) < 1e-9)
        {
            continue;
        }

        const double x = homogeneousPoints4D64.at<double>(0, i) / w;
        const double y = homogeneousPoints4D64.at<double>(1, i) / w;
        const double z = homogeneousPoints4D64.at<double>(2, i) / w;

        // Rejet des résultats non numériques ou infinis
        if (!std::isfinite(x) ||
            !std::isfinite(y) ||
            !std::isfinite(z))
        {
            continue;
        }

        // Point exprimé dans le repère de la 1ère caméra
        cv::Mat pointCamera1 = 
            (cv::Mat_<double>(3, 1) << x, y, z);

        // Même point dans le repère de la 2ème caméra : X2 = R * X1 + t
        cv::Mat pointCamera2 = 
            rotation * pointCamera1 + translation;

        const double depth1 = z;
        const double depth2 = pointCamera2.at<double>(2, 0);

        // Le point doit être devant les deux caméras
        if (depth1 <= 0.0 ||
            depth2 <= 0.0)
        {
            continue;
        }

        triangulatedPoints.emplace_back(
            x,
            y,
            z
        );

        validTriangulationPoints1.push_back(
            triangulationPoints1[i]
        );

        validTriangulationPoints2.push_back(
            triangulationPoints2[i]
        );
    }

    std::cout << "\n=== Triangulation ===\n\n";

    std::cout << "Input correspondences: "
              << triangulationPoints1.size()
              << '\n';

    std::cout << "Valid 3D points: "
              << triangulatedPoints.size()
              << '\n';

    const std::size_t pointsToPrint =
        std::min<std::size_t>(
            10,
            triangulatedPoints.size()
        );

    for (std::size_t i = 0; i < pointsToPrint; ++i)
    {
        const cv::Point3d& point = triangulatedPoints[i];

        std::cout << "Point "
                  << i
                  << ": ["
                  << point.x
                  << ", "
                  << point.y
                  << ", "
                  << point.z
                  << "]\n";
    }

    // -------------------------------------------------------
    // Calcul des erreurs de reprojection
    // 
    // Pour chaque point 3D reconstruit, on utilise P1 & P2 
    // pour prédire sa position dans chacune des images.
    // 
    // On compare ensuite cette prédiction aux coordonnées 2D 
    // réellement observées.
    // -------------------------------------------------------
    double totalSquaredReprojectionError = 0.0;
    double maximumReprojectionError = 0.0;

    for (std::size_t i = 0; i < triangulatedPoints.size(); ++i)
    {
        const cv::Point3d& point3D = 
            triangulatedPoints[i];


        // Coordonnées homogènes du point 3D :
        // 
        // [ X ]
        // [ Y ]
        // [ Z ]
        // [ 1 ]
        //
        cv::Mat homogeneousPoint =
            (cv::Mat_<double>(4, 1)
                << point3D.x,
                   point3D.y,
                   point3D.z,
                   1.0);

        // Projection dans la 1ère image
        cv::Mat projected1 =
            projectionMatrix1 * homogeneousPoint;

        // Projection dans la 2nde image
        cv::Mat projected2 =
            projectionMatrix2 * homogeneousPoint;

        // Passage :
        //
        // (uh, vh, wh)
        // 
        // vers
        // 
        // (u, v) = (uh / wh, vh / wh)
        const double u1 =
            projected1.at<double>(0, 0) / projected1.at<double>(2, 0);

        const double v1 = 
            projected1.at<double>(1, 0) / projected1.at<double>(2, 0);

        const double u2 =
            projected2.at<double>(0, 0) / projected2.at<double>(2, 0);

        const double v2 =
            projected2.at<double>(1, 0) / projected2.at<double>(2, 0);


        // Observations réellement détectées dans les images.
        const cv::Point2f& observed1 =
            validTriangulationPoints1[i];

        const cv::Point2f& observed2 =
            validTriangulationPoints2[i];


        // Distance euclidienne en pixels entre
        // projection prédite et observation réelle.
        const double error1 = 
            std::hypot(
                u1 - observed1.x,
                v1 - observed1.y
            );

        const double error2 =
            std::hypot(
                u2 - observed2.x,
                v2 - observed2.y
            );
        

        totalSquaredReprojectionError +=
            error1 * error1 +
            error2 * error2;

        maximumReprojectionError = 
            std::max(
                maximumReprojectionError,
                std::max(error1, error2)
            );

        std::cout << "Point "
                  << i
                  << " reprojection errors: "
                  << error1
                  << " px / "
                  << error2
                  << " px\n";
    }

    double reprojectionRms = 0.0;

    if (!triangulatedPoints.empty())
    {
        reprojectionRms = 
            std::sqrt(
                totalSquaredReprojectionError /
                (
                    2.0 * static_cast<double>(
                        triangulatedPoints.size()
                        )
                    )
            );
    }

    std::cout << "\n=== 3D reconstruction quality ===\n";

    std::cout << "Reprojection RMS: "
              << reprojectionRms
              << " px\n";

    std::cout << "Maximum reprojection error: "
              << maximumReprojectionError
              << " px\n";



    // --------------------------------------------------------
    // Visualisation
    // 
    // drawMatches place les deux images côte à côte et relie
    // chaque paire de points correspondants par une ligne.
    // --------------------------------------------------------
    cv::Mat visualization;

    cv::drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        poseInliersMatches,
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