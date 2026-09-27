#include <opencv2/core/utils/logger.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

#include <opencv2/opencv.hpp>


namespace fs = std::filesystem;

int addNewMapPointsBetweenKeyframes(
    const std::vector<cv::KeyPoint>& keyframeKeypoints,
    const cv::Mat& keyframeDescriptors,
    std::vector<int>& keyframeMapPointIndices,
    const cv::Mat& keyframeWorldToCameraRotation,
    const cv::Mat& keyframeTranslationVector,

    const std::vector<cv::KeyPoint>& currentKeypoints,
    const cv::Mat& currentDescriptors,
    std::vector<int>& currentMapPointIndices,
    const cv::Mat& currentWorldToCameraRotation,
    const cv::Mat& currentTranslationVector,

    const cv::Mat& cameraMatrix,
    const cv::Mat& distortionCoefficients,

    float ratioThreshold,

    std::vector<cv::Point3f>& mapPoints,
    cv::Mat& mapDescriptors)
{
    // -----------------------------------------------------------
    // Matching entre ancienne keyframe et nouvelle keyframe
    // 
    // On cherche ici des features qui ne correspondent PAS encore
    // à des points 3D existants
    // -----------------------------------------------------------
    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    std::vector<std::vector<cv::DMatch>> knnMatches;

    matcher.knnMatch(
        keyframeDescriptors,
        currentDescriptors,
        knnMatches,
        2
    );

    std::vector<cv::DMatch> candidateMatches;

    for (const auto& candidates : knnMatches)
    {
        if (candidates.size() < 2)
        {
            continue;
        }

        const cv::DMatch& best = candidates[0];
        const cv::DMatch& second = candidates[1];

        // Lowe ratio test
        if (best.distance >= ratioThreshold * second.distance)
        {
            continue;
        }

        // Le point de l'ancienne keyframe appartient déjà à la carte:
        // inutile de le trianguler à nouveau
        if (keyframeMapPointIndices[best.queryIdx] >= 0)
        {
            continue;
        }

        // Le point courant appartient déjà à la carte.
        if (currentMapPointIndices[best.trainIdx] >= 0)
        {
            continue;
        }

        candidateMatches.push_back(
            best
        );
    }
    // On privilégie les meilleurs matches avantt de suppimer les
    // duplications éventuelles sur l'image courante.
    std::sort(
        candidateMatches.begin(),
        candidateMatches.end(),
        [](const cv::DMatch& a,
            const cv::DMatch& b)
        {
            return a.distance < b.distance;
        }
    );

    std::vector<bool> currentKeypointAlreadyUsed(
        currentKeypoints.size(),
        false
    );

    std::vector<cv::DMatch> triangulationMatches;

    for (const cv::DMatch& match : candidateMatches)
    {
        if (currentKeypointAlreadyUsed[match.trainIdx])
        {
            continue;
        }

        currentKeypointAlreadyUsed[match.trainIdx] = true;

        triangulationMatches.push_back(match);
    }

    if (triangulationMatches.size() < 20)
    {
        return 0;    
    }

    // --------------------------------------------------------------
    // Coordonnées 2D correspondant aux nouveaux matches
    // --------------------------------------------------------------
    std::vector<cv::Point2f> pointsKeyframe;
    std::vector<cv::Point2f> pointsCurrent;

    for (const cv::DMatch& match : triangulationMatches)
    {
        pointsKeyframe.push_back(
            keyframeKeypoints[match.queryIdx].pt
        );
        pointsCurrent.push_back(
            currentKeypoints[match.trainIdx].pt
        );
    }

    // --------------------------------------------------------------
    // Correction de la distorsion
    // 
    // P = cameraMatrix signifie que les coordonnées résultantes
    // restent exprimées en pixels.
    // --------------------------------------------------------------
    std::vector<cv::Point2f> undistortedKeyframe;
    std::vector<cv::Point2f> undistortedCurrent;

    cv::undistortPoints(
        pointsKeyframe,
        undistortedKeyframe,
        cameraMatrix,
        distortionCoefficients,
        cv::noArray(),
        cameraMatrix
    );

    cv::undistortPoints(
        pointsCurrent,
        undistortedCurrent,
        cameraMatrix,
        distortionCoefficients,
        cv::noArray(),
        cameraMatrix
    );

    // ------------------------------------------------------------
    // Matrices de projection ABSOLUES
    // 
    // Contrairement au bootstrap initial:
    // 
    // P1 = K [ R1 | t1 ]
    // P2 = K [ R2 | t2 ]
    // 
    // Les points triangulés seront donc directement exprimés dans
    // le repère monde de notre carte.
    // ------------------------------------------------------------
    cv::Mat keyframeExtrinsic;
    cv::Mat currentExtrinsic;

    cv::hconcat(
        currentWorldToCameraRotation,
        currentTranslationVector,
        currentExtrinsic
    );

    cv::hconcat(
        keyframeWorldToCameraRotation,
        keyframeTranslationVector,
        keyframeExtrinsic
    );

    const cv::Mat projectionKeyframe = 
        cameraMatrix * keyframeExtrinsic;

    const cv::Mat projectionCurrent =
        cameraMatrix * currentExtrinsic;


    // ------------------------------------------------------------
    // Centres des deux caméras dans le repère monde
    // 
    // Pour une pose: X_camera = Rcw * X_world + tcw
    // 
    // Centre caméra dans le monde: Cw = -Rcw^T * tcw
    // ------------------------------------------------------------
    const cv::Mat keyframeCameraCenterWorld = 
        -keyframeWorldToCameraRotation.t() * keyframeTranslationVector;

    const cv::Mat currentCameraCenterWorld = 
        -currentWorldToCameraRotation.t() * currentTranslationVector;


    // ------------------------------------------------------------
    // Angle minimal de triangulation
    // 
    // Si les 2 rayons de vue sont presque parallèles, la 
    // profondeur du point devient très mal contrainte.
    // ------------------------------------------------------------
    const double minimumTriangulationAngleDegrees = 0.25;


    // ------------------------------------------------------------
    // Triangulation
    // ------------------------------------------------------------
    cv::Mat homogeneousPoints4D;

    cv::triangulatePoints(
        projectionKeyframe,
        projectionCurrent,
        undistortedKeyframe,
        undistortedCurrent,
        homogeneousPoints4D
    );

    cv::Mat homogeneousPoints4D64;
    homogeneousPoints4D.convertTo(
        homogeneousPoints4D64,
        CV_64F
    );


    std::vector<double> triangulationAngles;

    // -----------------------------------------------------------
    // Validation et ajout à la carte
    // 
    // On impose :
    //      - coordonnées finies;
    //      - profondeur poisitive dans les deux caméras;
    //      - erreur de reprojection <= 2 px dans les deux vues.
    // -----------------------------------------------------------
    const double maximumReprojectionError = 2.0;

    int addedPoints = 0;
    int rejectedByLowParallax = 0;

    for (int i = 0; i < homogeneousPoints4D64.cols; ++i) {
        const double w =
            homogeneousPoints4D64.at<double>(
                3,i
            );

        if (std::abs(w) < 1e-9) {
            continue;
        }

        const double pointX = 
            homogeneousPoints4D64.at<double>(
                0, i
            ) / w;

        const double pointY =
            homogeneousPoints4D64.at<double>(
                1, i
            ) / w;

        const double pointZ =
            homogeneousPoints4D64.at<double>(
                2, i
            ) / w;

        if (!std::isfinite(pointX) ||
            !std::isfinite(pointY) ||
            !std::isfinite(pointZ))
        {
            continue;
        }

        cv::Mat pointWorld = 
            (cv::Mat_<double>(3, 1)
                << pointX,
                   pointY,
                   pointZ);

        cv::Mat pointKeyframe = 
            keyframeWorldToCameraRotation * pointWorld + keyframeTranslationVector;

        cv::Mat pointCurrent =
            currentWorldToCameraRotation * pointWorld + currentTranslationVector;

        // Point derrière une caméra
        if (pointKeyframe.at<double>(2, 0) <= 0.0 ||
            pointCurrent.at<double>(2, 0) <= 0.0)
        {
            continue;
        }
        
        // --------------------------------------------------------
        // Contrôle de parallaxe
        //
        // On construit les deux rayons allant des centres caméra
        // vers le point 3D.
        //
        //                 X
        //                / \
        //               /   \
        //             C1     C2
        //
        // L'angle entre ces deux rayons doit être suffisamment
        // grand pour que la profondeur soit bien contrainte.
        // --------------------------------------------------------
        cv::Mat rayFromKeyframe = 
            pointWorld - keyframeCameraCenterWorld;

        cv::Mat rayFromCurrent =
            pointWorld - currentCameraCenterWorld;

        const double keyframeRayNorm = 
            cv::norm(rayFromKeyframe);

        const double currentRayNorm =
            cv::norm(rayFromCurrent);

        // Protection contre les cas dégénérés
        if (keyframeRayNorm < 1e-9 ||
            currentRayNorm < 1e-9)
        {
            continue;
        }

        // Le cosinus d'un angle theta est donné par:
        // cos(theta) = (a.b) / (||a||  ||b||)
        double cosineAngle = 
            rayFromKeyframe.dot(rayFromCurrent) / (keyframeRayNorm * currentRayNorm);

        // Les petites erreurs numériques peuvent produire par exemple
        // 1.0000000001, ce qui rendrait acos() invalide.
        cosineAngle = 
            std::clamp(
                cosineAngle,
                -1.0,
                1.0
            );

        const double triangulationAngleDegrees =
            std::acos(cosineAngle) * 180.0 / CV_PI;


        triangulationAngles.push_back(
            triangulationAngleDegrees
        );

        if (triangulationAngleDegrees < minimumTriangulationAngleDegrees)
        {
            ++rejectedByLowParallax;
            continue;
        }





        // -------------------------------------------------------
        // Reprojection dans la keyframe
        // -------------------------------------------------------
        cv::Mat projectedKeyframe = cameraMatrix * pointKeyframe;

        const cv::Point2d reprojectionKeyframe(
            projectedKeyframe.at<double>(0, 0) / projectedKeyframe.at<double>(2, 0),
            projectedKeyframe.at<double>(1, 0) / projectedKeyframe.at<double>(2, 0)
        );

        // -------------------------------------------------------
        // Reprojection dans la frame courante
        // -------------------------------------------------------
        cv::Mat projectedCurrent = cameraMatrix * pointCurrent;

        const cv::Point2d reprojectionCurrent(
            projectedCurrent.at<double>(0, 0) / projectedCurrent.at<double>(2, 0),
            projectedCurrent.at<double>(1, 0) / projectedCurrent.at<double>(2, 0)
        );


        const double errorKeyframe = 
            cv::norm(
                reprojectionKeyframe - cv::Point2d(undistortedKeyframe[i])
            );

        const double errorCurrent =
            cv::norm(
                reprojectionCurrent - cv::Point2d(undistortedCurrent[i])
            );


        if (errorKeyframe > maximumReprojectionError ||
            errorCurrent > maximumReprojectionError)
        {
            continue;
        }


        // -------------------------------------------------------
        // Nouveau point valide
        // -------------------------------------------------------
        const cv::DMatch& match = triangulationMatches[i];
        const int newMapPointIndex = 
            static_cast<int>(
                mapPoints.size()
            );

        mapPoints.emplace_back(
            static_cast<float>(pointX),
            static_cast<float>(pointY),
            static_cast<float>(pointZ)
        );

        // On conserve le descripteur de l'observation la plus récente.
        mapDescriptors.push_back(
            currentDescriptors.row(match.trainIdx)
        );

        // Association dans les deux keyframes
        keyframeMapPointIndices[match.queryIdx] = newMapPointIndex;
        currentMapPointIndices[match.trainIdx] = newMapPointIndex;

        ++addedPoints;
    }

    std::cout << "[TRIANGULATION] "
              << "matches: "
              << triangulationMatches.size()
              << " | low-parallax rejected: "
              << rejectedByLowParallax
              << " | added: "
              << addedPoints
              << '\n';


    if (!triangulationAngles.empty())
    {
        std::sort(
            triangulationAngles.begin(),
            triangulationAngles.end()
        );

        const double minimumAngle = 
            triangulationAngles.front();

        const double medianAngle = 
            triangulationAngles[triangulationAngles.size() / 2];

        const double maximumAngle =
            triangulationAngles.back();

        int above025 = 0;
        int above050 = 0;
        int above100 = 0;

        for (double angle : triangulationAngles) 
        {
            if (angle >= 0.25)
            {
                ++above025;
            }

            if (angle >= 0.50)
            {
                ++above050;
            }

            if (angle >= 1.00)
            {
                ++above100;
            }
        }

        std::cout << "[PARALLAX] min: "
                  << minimumAngle
                  << " deg"
                  << " | median: "
                  << medianAngle
                  << " | max: "
                  << maximumAngle
                  << " deg\n";

        std::cout << "[PARALLAX] >=0.25: "
                  << above025
                  << " | >=0.50: "
                  << above050
                  << " | >= 1.00: "
                  << above100
                  << '\n';
    }

    return addedPoints;
}



int main(int argc, char* argv[])
{
    // ------------------------------------------------------------
    // Réduit les messages internes d'OpenCV.
    //
    // On garde uniquement les vraies erreurs OpenCV.
    // Les messages INFO concernant FFMPEG, GStreamer, TBB,
    // MSMF, etc. ne seront plus affichés.
    // ------------------------------------------------------------
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_ERROR
    );

    
	std::cout << "\n=== Video Monocular Visual Odometry ===\n\n";

    // ------------------------------------------------------------
    // Arguments
    //
    // argv[1] : vidéo
    // argv[2] : calibration caméra
    // argv[3] : fichier CSV de trajectoire
    // argv[4] : intervalle temporel entre deux images utilisées
    //
    // Exemple :
    //
    // map_visual_odometry
    //     sequence_main.mp4
    //     camera_calibration_video_1920x1080.yaml
    //     trajectory.csv
    //     0.5
    // ------------------------------------------------------------
    if (argc < 4)
    {
        std::cerr << "Usage: video_visual_odometry "
                  << "<video_path> "
                  << "<calibration_file> "
                  << "<output_csv> "
                  << "[sampling interval seconds]\n";

        return 1;
    }

    const fs::path videoPath = argv[1];
    const fs::path calibrationPath = argv[2];
    const fs::path outputCsvPath = argv[3];

    // 0.5 seconde par défaut
    double samplingIntervalSeconds = 0.5;

    if (argc >= 5)
    {
        try
        {
            samplingIntervalSeconds = std::stod(argv[4]);
        }
        catch (...)
        {
            std::cerr << "Error: invalid sampling interval.\n";
            return 1;
        }
    }

    if (samplingIntervalSeconds <= 0.0)
    {
        std::cerr << "Error: sampling interval must be positive.\n";
        return 1;
    }


    // ------------------------------------------------------------
    // Chargement de la calibration vidéo
    // ------------------------------------------------------------
    cv::FileStorage calibrationFile(
        calibrationPath.string(),
        cv::FileStorage::READ
    );


    if (!calibrationFile.isOpened()) 
    {
        std::cerr << "Error: could not open calibration file: "
                  << calibrationPath
                  << '\n';

        return 1;
    }

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

    if (cameraMatrix.empty() ||
        distortionCoefficients.empty() ||
        cameraMatrix.rows != 3 ||
        cameraMatrix.cols != 3)
    {
        std::cerr << "Error: invalid camera calibration.\n";
        return 1;
    }

    std::cout << "Calibration resolution: "
              << calibrationWidth
              << " x "
              << calibrationHeight
              << "\n\n";


    // ------------------------------------------------------------
    // Ouverture de la vidéo
    // ------------------------------------------------------------
    cv::VideoCapture video(
        videoPath.string()
    );


    if (!video.isOpened())
    {
        std::cerr << "Error: could not open video: "
                  << videoPath
                  << '\n';

        return 1;
    }

    const double fps = video.get(cv::CAP_PROP_FPS);
    const int frameCount = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_COUNT)
    );

    const int videoWidth = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_WIDTH)
    );

    const int videoHeight = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_HEIGHT)
    );


    if (fps <= 0.0)
    {
        std::cerr << "Error: invalide video FPS.\n";
        return 1;
    }


    std::cout << "Video resolution: "
              << videoWidth
              << " x "
              << videoHeight
              << '\n';

    std:: cout << "FPS: "
               << fps
               << '\n';

    std::cout << "Frame count: "
              << frameCount
              << '\n';

    // La vidéo de calibration et la vidéo de travail 
    // doivent être de même résolution.
    if (videoWidth != calibrationWidth ||
        videoHeight != calibrationHeight)
    {
        std::cerr << "Error: video resolution does not match "
                  << "the calibration resolution.\n";

        return 1;
    }

    // Conversion de l'intervalle temporel en nombre de frames
    // 
    // Exemple:
    //  20 fps * 0.5 s = 10 frames
    const int frameStep = std::max(
        1, static_cast<int>(
                std::round(
                    fps * samplingIntervalSeconds
                )
            )
    );

    std::cout << "Sampling interval: "
              << samplingIntervalSeconds
              << " s\n";

    std::cout << "Frame step: "
              << frameStep
              << "\n\n";


    // ------------------------------------------------------------
    // Préparation du fichier de trajectoire
    // ------------------------------------------------------------
    if (outputCsvPath.has_parent_path())
    {
        fs::create_directories(
            outputCsvPath.parent_path()
        );
    }

    std::ofstream trajectoryFile(
        outputCsvPath
    );

    if (!trajectoryFile.is_open())
    {
        std::cerr << "Error: could not create trajectory file.\n";
        return 1;
    }


    trajectoryFile << "frame_index,"
                   << "time_seconds,"
                   << "x, "
                   << "y, "
                   << "z, "
                   << "good_matches,"
                   << "pose_inliers\n";


    // ------------------------------------------------------------
    // ORB + matcher
    // 
    // On augmente à 2000 pts par rapport au mode photo car les 
    // vidéos sont en 1920x1080. 
    // ------------------------------------------------------------
    cv::Ptr<cv::ORB> orb =
        cv::ORB::create(2000);

    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    const float ratioThreshold = 0.75f;


    // -------------------------------------------------------------
    // Critères minimaux pour accepter une pose PnP
    // 
    //  - au moins 20 correspondances 3D-2D avant PnP;
    //  - au moins 20 inliers après RANSAC;
    //  - au moins 40 % des correspondances doivent être des inliers.
    // 
    // Le ratio évite par exemple d'accepter une pose avec 20 inliers
    // sur 200 correspondances.
    // -------------------------------------------------------------
    const int minimumPnPCorrespondences = 20;
    const int minimumPnPInliers = 20;

    const double minimumPnPInlierRatio = 0.40;



    // ------------------------------------------------------------
    // Carte 3D persistante
    // 
    // Invariant important:
    // mapPoints[i] correspond toujours à mapDescriptors.row(i)
    // 
    // Chaque ligne de descripteur décrit donc le même point 
    // physique que la position 3D de même indice.
    // ------------------------------------------------------------
    std::vector<cv::Point3f> mapPoints;
    cv::Mat mapDescriptors;
    bool mapInitialized = false;

    // ------------------------------------------------------------
    // Keyframe courante
    // Une keyframe est une image de référence dont on conserve:
    //      - les keypoints ORB;
    //      - les descripteurs;
    //      - les associations keypoint -> point 3D de la carte;
    //      - sa pose absolue dans le repère monde.
    // 
    // keyframePointIndices[i] = -1
    //      -> le keypoint i n'est associé à aucun point 3D.
    // keyframePointIndices[i] >= 0
    //      -> indice du point correspondant dans mapPoints.
    // ------------------------------------------------------------
    std::vector<cv::KeyPoint> keyframeKeypoints;
    cv::Mat keyframeDescriptors;
    std::vector<int> keyframeMapPointIndices;

    // Pose monde -> caméra de la keyframe:
    //      X_camera = Rcw * X_world + tcw
    cv::Mat keyframeWorldToCameraRotation;
    cv::Mat keyframeTranslationVector;

    int keyframeFrameIndex = -1;


    // ----------------------------------------------------------------
    // Politique provisoire de création des keyframes.
    // On envisage une nouvelle keyframe lorsque:
    //      1. suffisamment de temps s'est écoulé depuis la précédente;
    //      2. le nombre d'inliers PnP commence à diminuer.
    // 
    // 3 * frameStep = environ 1.5 seconde ici.
    // ----------------------------------------------------------------
    const int minimumKeyframeGapFrames = 3 * frameStep;
    const int keyframeInlierThreshold = 120;
    
    // ------------------------------------------------------------
    // Etat de la frame précédente acceptée
    // ------------------------------------------------------------
    cv::Mat previousGray;
    cv::Mat previousDescriptors;

    std::vector<cv::KeyPoint> previousKeypoints;

    bool previousFrameInitialized = false;


    // ------------------------------------------------------------
    // Pose globale de la caméra
    // 
    // Rwc : rotation du repère caméra vers le repère monde.
    // Cw : position du centre de la caméra dans le repère monde.
    // 
    // La première caméra définit notre origine: 
    //      Rwc = I et Cw = 0
    // ------------------------------------------------------------
    cv::Mat worldFromCameraRotation = 
        cv::Mat::eye(
            3,
            3,
            CV_64F
        );

    cv::Mat cameraPositionWorld =
        cv::Mat::zeros(
            3,
            1,
            CV_64F
        );


    // ------------------------------------------------------------
    // Historique utilisé pour vérifier la cohérence du mouvement
    // 
    // lastAcceptedPoseFrameIndex : 
    //      frame correspondant à cameraPositionWorld
    // 
    // recentNormalizedStepDistances :
    //      derniers déplacements acceptés, ramenés à un intervalle 
    //      normal de frameStep
    // ------------------------------------------------------------
    int lastAcceptedPoseFrameIndex = -1;

    std::vector<double> recentNormalizedStepDistances;


    // Nombre de déplacements récents utilisés pour estimer 
    // le mouvement "normal"
    const std::size_t motionHistorySize = 5;

    // On autorise un deplacement jusqu'à 3 fois la médiane 
    // récente avant de le considérer comme suspect.
    const double maximumStepMedianMultiplier = 3.0;

    // Sécurité pour éviter qu'une période quasi immobile
    // rende le seuil exceesivement petit
    // 
    // ATTENTION: ce sont toujours des unités arbitraires
    // monoculaires, pas des mètres
    const double minimumMaximumNormalizedStep = 0.5;


        
    // ------------------------------------------------------------
    // Lecture séquentielle de la vidéo
    // ------------------------------------------------------------
    cv::Mat frame;

    int frameIndex = 0;
    int selectedFrames = 0;
    int successfulPoseSteps = 0;

    while (true)
    {
        const bool frameRead =
            video.read(frame);

        if (!frameRead)
        {
            std::cout
                << "\n[VIDEO] Reading stopped at frame "
                << frameIndex
                << " / "
                << frameCount
                << '\n';

            break;
        }

        // On ne traite qu'une frame sur frameStep.
        if (frameIndex % frameStep != 0)
        {
            ++frameIndex;
            continue;
        }

        ++selectedFrames;

        cv::Mat gray;
        cv::cvtColor(
            frame, 
            gray,
            cv::COLOR_BGR2GRAY
        );


        // Détection ORB sur la frame courante
        std::vector<cv::KeyPoint> currentKeypoints;
        cv::Mat currentDescriptors;

        orb->detectAndCompute(
            gray,
            cv::noArray(),
            currentKeypoints,
            currentDescriptors
        );

        if (currentDescriptors.empty())
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : no descriptors\n";

            ++frameIndex;
            continue;
        }

        // Première frame: simple initialisation de la trajectoire
        if (!previousFrameInitialized)
        {
            previousGray = gray.clone();
            previousKeypoints = currentKeypoints;
            previousDescriptors = currentDescriptors.clone();
            previousFrameInitialized = true;

            trajectoryFile << frameIndex
                           << ","
                           << static_cast<double>(frameIndex) / fps
                           << ",0,0,0,0,0\n";

            std::cout << "[INIT] frame "
                      << frameIndex
                      << " : "
                      << currentKeypoints.size()
                      << " keypoints\n";

            ++frameIndex;
            continue;
        }

        // ---------------------------------------------------------------
        // Localisation par rapport à la carte 3D existante
        // ---------------------------------------------------------------
        if (mapInitialized)
        {
            // Matching :
            //
            // mapDescriptors       = descripteurs associés aux points 3D
            // currentDescriptors   = descripteurs de l'image actuelle
            //
            // queryIdx -> indice dans la carte
            // trainIdx -> indice dans currentKeypoints
            std::vector<std::vector<cv::DMatch>> mapKnnMatches;

            matcher.knnMatch(
                mapDescriptors,
                currentDescriptors,
                mapKnnMatches,
                2
            );


            std::vector<cv::DMatch> mapGoodMatches;
            
            for (const auto& candidates : mapKnnMatches)
            {
                if (candidates.size() < 2)
                {
                    continue;
                }

                if (candidates[0].distance < ratioThreshold * candidates[1].distance)
                {
                    mapGoodMatches.push_back(candidates[0]);
                }
            }

            // Construction des correspondances:
            // point 3D de la carte <-> point 2D de l'image
            std::vector<cv::Point3f> pnpObjectPoints;
            std::vector<cv::Point2f> pnpImagePoints;

            pnpObjectPoints.reserve(
                mapGoodMatches.size()
            );
            pnpImagePoints.reserve(
                mapGoodMatches.size()
            );

            for (const cv::DMatch& match : mapGoodMatches)
            {
                pnpObjectPoints.push_back(
                    mapPoints[match.queryIdx]
                );

                pnpImagePoints.push_back(
                    currentKeypoints[match.trainIdx].pt
                );
            }


            // Cas où il y a pas assez de correspondances 3D-2D: 
            // impossible de localiser proprement la caméra.
            if (pnpObjectPoints.size() < minimumPnPCorrespondences)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : only "
                          << pnpObjectPoints.size()
                          << " 3D-2D matches\n";

                ++frameIndex;
                continue;
            }

            // Estimation robuste de la pose
            cv::Mat rotationVector;
            cv::Mat translationVector;
            cv::Mat pnpInliers;

            const bool pnpSuccess = 
                cv::solvePnPRansac(
                    pnpObjectPoints,
                    pnpImagePoints,
                    cameraMatrix,
                    distortionCoefficients,
                    rotationVector,
                    translationVector,
                    false,
                    100,
                    3.0,
                    0.99,
                    pnpInliers,
                    cv::SOLVEPNP_ITERATIVE
                );


            // ------------------------------------------------------
            // Validation de la pose PnP
            // ------------------------------------------------------

            // 1ère condition : solvePnPRansac doit avoir réussi
            if (!pnpSuccess)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : solvePnPRansac failed\n";

                ++frameIndex;
                continue;
            }

            // Proportion des correspondances 3D-2D qui ont réellement 
            // été considérées comme cohérentes par RANSAC.
            // Exemple: 80 matches, 60 inliers -> ratio = 60/80 = 0.75
            const double pnpInlierRatio =
                static_cast<double>(pnpInliers.rows) / static_cast<double>(pnpObjectPoints.size());

            // 2ème condition : quantité absolue suffisante d'inliers
            if (pnpInliers.rows < minimumPnPInliers)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : "
                          << pnpInliers.rows
                          << " PnP inliers"
                          << " | ratio: "
                          << pnpInlierRatio
                          << '\n';

                ++frameIndex;
                continue;
            }

            // 3ème condition : proportion suffisante d'inliers.
            // Une pose peut avoir 20 ou 30 inliers mais rester douteuse
            // si elle avait plusieurs centaines de correspondances
            // candidates au départ.
            if (pnpInlierRatio < minimumPnPInlierRatio)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : low inlier ratio"
                          << " | matches: "
                          << pnpObjectPoints.size()
                          << " | inliers: "
                          << pnpInliers.rows
                          << " | ratio: "
                          << pnpInlierRatio
                          << '\n';

                ++frameIndex;
                continue;
            }


            
            // solvePnP donne: X_camera = Rcw * X_world + tcw
            // Position caméra dans le monde: Cw = -Rcw^T * tcw
            cv::Mat worldToCameraRotation;
            cv::Rodrigues(
                rotationVector,
                worldToCameraRotation
            );

            // --------------------------------------------------------
            // Association des keypoints de la frmae courante avec les 
            // points 3D réellement validés par PnP/RANSAC.
            // --------------------------------------------------------
            std::vector<int> currentMapPointIndices(
                currentKeypoints.size(), -1
            );

            for (int i = 0; i < pnpInliers.rows; ++i)
            {
                // solvePnPRansac retourne les indices dans pnpObjectPoints /
                // pnpImagePoints.
                // Ces deux vecteurs ont exactement le même ordre que
                // mapGoodMatches.
                const int matchListIndex =
                    pnpInliers.at<int>(
                        i, 0
                    );

                const cv::DMatch& inlierMatch =
                    mapGoodMatches[matchListIndex];

                // queryIdx = point de carte
                // trainIdx = keypoint image courante
                currentMapPointIndices[inlierMatch.trainIdx] = inlierMatch.queryIdx;
            }

            const cv::Mat candidateCameraPositionWorld = 
                -worldToCameraRotation.t() * translationVector;

            // -----------------------------------------------------------
            // Cohérence temporelle de la nouvelle pose
            // -----------------------------------------------------------
            const int frameGap = 
                frameIndex - lastAcceptedPoseFrameIndex;

            // Distance 3D entre la dernière pose acceptée et la nouvelle
            // pose proposée par PnP
            const double rawStepDistance = 
                cv::norm(
                    candidateCameraPositionWorld - cameraPositionWorld
                );

            // Si une ou plusieurs frames ont été rejetées,
            // rawStepDistance correspond à un temps plus long.
            //
            // On ramène donc toujours le déplacement à un intervalle
            // standard de frameStep.
            //
            // Exemple :
            //     frameStep = 15
            //
            // 15 frames écoulées : facteur 1
            // 30 frames écoulées : facteur 0.5
            // 45 frames écoulées : facteur 1/3
            const double normalizedStepDistance = 
                rawStepDistance * 
                static_cast<double>(frameStep) /
                static_cast<double>(frameGap);
                

            double recentMedianStep = 0.0;
            double maximumAllowedStep =
                std::numeric_limits<double>::infinity();

            if (recentNormalizedStepDistances.size() >=
                motionHistorySize) 
            {
                std::vector<double> sortedRecentSteps = 
                    recentNormalizedStepDistances;

                std::sort(
                    sortedRecentSteps.begin(),
                    sortedRecentSteps.end()
                );

                recentMedianStep =
                    sortedRecentSteps[
                        sortedRecentSteps.size() / 2
                    ];

                maximumAllowedStep =
                    std::max(
                        minimumMaximumNormalizedStep,
                        maximumStepMedianMultiplier * recentMedianStep
                    );
            }

            if (recentNormalizedStepDistances.size() >=
                motionHistorySize &&
                normalizedStepDistance >
                maximumAllowedStep)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : motion jump"
                          << " | step: "
                          << normalizedStepDistance
                          << " | recent median: "
                          << recentMedianStep
                          << " | allowed: "
                          << maximumAllowedStep
                          << '\n';

                ++frameIndex;
                continue;
            }

            cameraPositionWorld =
                candidateCameraPositionWorld.clone();

            worldFromCameraRotation = worldToCameraRotation.t();

            lastAcceptedPoseFrameIndex =
                frameIndex;

            recentNormalizedStepDistances.push_back(
                normalizedStepDistance
            );

            if (recentNormalizedStepDistances.size() > motionHistorySize)
            {
                recentNormalizedStepDistances.erase(
                    recentNormalizedStepDistances.begin()
                );
            }

            ++successfulPoseSteps;

            const double x = cameraPositionWorld.at<double>(0, 0);
            const double y = cameraPositionWorld.at<double>(1, 0);
            const double z = cameraPositionWorld.at<double>(2, 0);

            // -----------------------------------------------------------
            // Décision de création d'une nouvelle keyframe
            // -----------------------------------------------------------
            const int framesSinceLastKeyframe = 
                frameIndex - keyframeFrameIndex;

            const bool enoughFrameSeparation = 
                framesSinceLastKeyframe >= minimumKeyframeGapFrames;

            const bool mapVisibilityIsDecreasing = 
                pnpInliers.rows < keyframeInlierThreshold;

            const bool shouldCreateKeyframe = 
                enoughFrameSeparation && mapVisibilityIsDecreasing;

            if (shouldCreateKeyframe)
            {
                std::cout << "[KEYFRAME] frame "
                          << frameIndex
                          << " selected"
                          << " | PnP inliers: "
                          << pnpInliers.rows
                          << " | previous keyframe: "
                          << keyframeFrameIndex
                          << '\n';

                // --------------------------------------------------------
                // Extension de la carte
                // 
                // IMPORTANT :
                // ceci doit être fait AVANT de remplacer l'ancienne
                // keyframe par la frame courante.
                // --------------------------------------------------------
                const std::size_t mapSizeBefore = mapPoints.size();

                const int addedMapPoints = 
                    addNewMapPointsBetweenKeyframes(
                        keyframeKeypoints,
                        keyframeDescriptors,
                        keyframeMapPointIndices,
                        keyframeWorldToCameraRotation,
                        keyframeTranslationVector,

                        currentKeypoints,
                        currentDescriptors,
                        currentMapPointIndices,
                        worldToCameraRotation,
                        translationVector,

                        cameraMatrix,
                        distortionCoefficients,

                        ratioThreshold,

                        mapPoints,
                        mapDescriptors
                    );
                
                std::cout << "[MAP] Added "
                          << addedMapPoints
                          << " new points"
                          << " | size: "
                          << mapSizeBefore
                          << " -> "
                          << mapPoints.size()
                          << '\n';


                // Verif permanente de notre invariant
                if (mapPoints.size() !=
                    static_cast<std::size_t>(
                        mapDescriptors.rows
                        ))
                {
                    std::cerr << "Error: mappoint/descriptor mismatch "
                              << "after map extension.\n";

                    return 1;
                }

                // La frame courante devient maintenant la nouvelle keyframe
                keyframeFrameIndex = frameIndex;
                keyframeKeypoints = currentKeypoints;
                keyframeDescriptors = currentDescriptors.clone();
                keyframeMapPointIndices = currentMapPointIndices;
                keyframeWorldToCameraRotation = worldToCameraRotation.clone();
                keyframeTranslationVector = translationVector.clone();

            }

            // Sauvegarde CSV
            trajectoryFile << frameIndex
                           << ","
                           << static_cast<double>(frameIndex) / fps
                           << ","
                           << x
                           << ","
                           << y
                           << ","
                           << z
                           << ","
                           << mapGoodMatches.size()
                           << ","
                           << pnpInliers.rows
                           << '\n';

            std::cout << "[PNP OK] frame "
                      << frameIndex
                      << " | map matches: "
                      << mapGoodMatches.size()
                      << " | PnP inliers: "
                      << pnpInliers.rows
                      << " | ratio: "
                      << pnpInlierRatio
                      << " | position: ["
                      << x
                      << ","
                      << y
                      << ","
                      << z
                      << "]\n";


            // Uitle pour la suite (pour l'extension dynamique de carte)
            previousGray = gray.clone();
            previousKeypoints = currentKeypoints;
            previousDescriptors = currentDescriptors.clone();

            ++frameIndex;

            // Très important: une fois PnP exécuté, on ne doit PAS 
            // continuer dans le pipeline Essentiel Matrix situé plus bas.
            continue;
        }

        // ---------------------------------------------------------------
        // Matching entre la dernière frame acceptée et la frame courante
        // ---------------------------------------------------------------
        std::vector<std::vector<cv::DMatch>> knnMatches;

        matcher.knnMatch(
            previousDescriptors,
            currentDescriptors,
            knnMatches,
            2
        );


        std::vector<cv::DMatch> goodMatches;

        for (const auto& candidates : knnMatches)
        {
            if (candidates.size() < 2)
            {
                continue;
            }

            if (candidates[0].distance < ratioThreshold * candidates[1].distance)
            {
                goodMatches.push_back(
                    candidates[0]
                );
            }
        }


        // On demande un minimum de correspondances pour éviter 
        // une estimation géométrique extrêmement fragile.
        if (goodMatches.size() < 20)
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : only "
                      << goodMatches.size()
                      << " good matches\n";

            ++frameIndex;
            continue;
        }

        // ------------------------------------------------------------
        // Coordonnées des correspondances
        // ------------------------------------------------------------
        std::vector<cv::Point2f> pointsPrevious;
        std::vector<cv::Point2f> pointsCurrent;


        pointsPrevious.reserve(
            goodMatches.size()
        );

        pointsCurrent.reserve(
            goodMatches.size()
        );


        for (const cv::DMatch& match : goodMatches)
        {
            pointsPrevious.push_back(
                previousKeypoints[match.queryIdx].pt
            );

            pointsCurrent.push_back(
                currentKeypoints[match.trainIdx].pt
            );
        }

        // ------------------------------------------------------------
        // Correction de la distorsion
        // ------------------------------------------------------------
        std::vector<cv::Point2f> undistortedPrevious;
        std::vector<cv::Point2f> undistortedCurrent;

        cv::undistortPoints(
            pointsPrevious,
            undistortedPrevious,
            cameraMatrix,
            distortionCoefficients,
            cv::noArray(),
            cameraMatrix
        );


        cv::undistortPoints(
            pointsCurrent,
            undistortedCurrent,
            cameraMatrix,
            distortionCoefficients,
            cv::noArray(),
            cameraMatrix
        );


        // -------------------------------------------------------------
        // Matrice essentielle
        // 
        // Contrairement au démonstrateur à 2 vues, on ne recalculera
        // pas F.
        // 
        // La caméra étant calibrée, findEssentialMat() fait directement
        // le RANSAC géométrique dont on a besoin pour la VO.
        // -------------------------------------------------------------
        cv::Mat essentialMask;

        cv::Mat essentialMatrix =
            cv::findEssentialMat(
                undistortedPrevious,
                undistortedCurrent,
                cameraMatrix,
                cv::RANSAC,
                0.999,
                1.0,
                essentialMask
            );

        if (essentialMatrix.empty())
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : essential matrix failed\n";

            ++frameIndex;
            continue;
        }



        // -------------------------------------------------------------
        // Pose relative
        // -------------------------------------------------------------
        cv::Mat relativeRotation;
        cv::Mat relativeTranslation;
        cv::Mat poseMask = essentialMask.clone();

        const int poseInliers = 
            cv::recoverPose(
                essentialMatrix,
                undistortedPrevious,
                undistortedCurrent,
                cameraMatrix,
                relativeRotation,
                relativeTranslation,
                poseMask
            );

        if (poseInliers < 15)
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : only "
                      << poseInliers
                      << " pose inliers\n";

            ++frameIndex;
            continue;
        }


        // -------------------------------------------------------------
        // Initialisation de la carte
        // 
        // Ce bloc ne doit être exécuté qu'une et une seule fois.
        // -------------------------------------------------------------
        if (!mapInitialized)
        {
            // Correspondances qui ont ont réellement survécu à recoverPose
            std::vector<cv::Point2f> bootstrapPoints1;
            std::vector<cv::Point2f> bootstrapPoints2;

            std::vector<cv::DMatch> bootstrapMatches;

            bootstrapPoints1.reserve(
                poseInliers
            );

            bootstrapPoints2.reserve(
                poseInliers
            );

            bootstrapMatches.reserve(
                poseInliers
            );


            for (std::size_t i = 0; i < goodMatches.size(); ++i)
            {
                if (poseMask.at<uchar>(static_cast<int>(i)) == 0)
                {
                    continue;
                }

                bootstrapPoints1.push_back(
                    undistortedPrevious[i]
                );

                bootstrapPoints2.push_back(
                    undistortedCurrent[i]
                );

                bootstrapMatches.push_back(
                    goodMatches[i]
                );
            }

            // Première caméra: P0 = K [ I | 0 ]
            cv::Mat firstCameraPose = 
                cv::Mat::zeros(
                    3, 
                    4,
                    CV_64F
                );

            cv::Mat identity = 
                cv::Mat::eye(
                    3, 3, CV_64F
                );
                
            identity.copyTo(
                firstCameraPose(
                    cv::Rect(0, 0, 3, 3)
                    )
                );


            // Deuxième caméra: P1 = K [ R | t ]
            cv::Mat secondCameraPose;

            cv::hconcat(
                relativeRotation,
                relativeTranslation,
                secondCameraPose
            );

            const cv::Mat projectionMatrix1 = cameraMatrix * firstCameraPose;
            const cv::Mat projectionMatrix2 = cameraMatrix * secondCameraPose;


            // Triangulation
            cv::Mat homogeneousPoints4D;

            cv::triangulatePoints(
                projectionMatrix1,
                projectionMatrix2,
                bootstrapPoints1,
                bootstrapPoints2,
                homogeneousPoints4D
            );

            cv::Mat homogeneousPoints4D64;

            homogeneousPoints4D.convertTo(
                homogeneousPoints4D64,
                CV_64F
            );

            // Nettoyage de tout éventuelle donnée précédente.
            // Normalement inutile au premier passage, mais explicite
            mapPoints.clear();
            mapDescriptors.release();
            
            // Pour chaque keypoint de la deuxième image du bootstrap,
            // mémorisation éventuelle du point 3D correspondant.
            std::vector<int> bootstrapMapPointIndices(
                currentKeypoints.size(),
                -1
            );

            // Conversion en points 3D euclidiens + cheirality
            for (int i = 0; i < homogeneousPoints4D64.cols; ++i)
            {
                const double w = homogeneousPoints4D64.at<double>(
                    3, i
                );

                if (std::abs(w) < 1e-9)
                {
                    continue;
                };

                const double pointX = homogeneousPoints4D64.at<double>(
                    0, i
                ) / w;

                const double pointY = homogeneousPoints4D64.at<double>(
                    1, i
                ) / w;

                const double pointZ = homogeneousPoints4D64.at<double>(
                    2, i
                ) / w;

                if (!std::isfinite(pointX) ||
                    !std::isfinite(pointY) ||
                    !std::isfinite(pointZ))
                {
                    continue;
                }

                cv::Mat pointCamera1 = 
                    (cv::Mat_<double>(3, 1)
                        << pointX, pointY, pointZ);

                cv::Mat pointCamera2 =
                    relativeRotation * pointCamera1 + relativeTranslation;

                const double depth1 = pointZ;
                const double depth2 = pointCamera2.at<double>(
                    2, 0
                );

                if (depth1 <= 0.0 ||
                    depth2 <= 0.0)
                {
                    continue;
                }

                // Correspondance qui a produit ce point 3D
                const cv::DMatch& match = bootstrapMatches[i];

                // Indice que le nouveau point va recevoir dans mapPoints
                const int newMapPointIndex = 
                    static_cast<int>(
                        mapPoints.size()
                    );

                mapPoints.emplace_back(
                    static_cast<float>(pointX),
                    static_cast<float>(pointY),
                    static_cast<float>(pointZ)
                );

                mapDescriptors.push_back(
                    currentDescriptors.row(
                        match.trainIdx
                    )
                );

                // Le keypoint de la deuxième image sait maintenant 
                // à quel point 3D il correspond.
                bootstrapMapPointIndices[match.trainIdx] = newMapPointIndex;
            }

            // Vérification de l'invariant carte/descripteurs
            if (mapPoints.size() !=
                static_cast<std::size_t>(
                    mapDescriptors.rows
                ))
            {
                std::cerr << "Error: map point/descriptor mismatch.\n";
                return 1;
            }

            if (mapPoints.size() < 20)
            {
                std::cout << "[MAP INIT FAILED] only "
                          << mapPoints.size()
                          << " valid 3D points\n";

                mapPoints.clear();
                mapDescriptors.release();

                ++frameIndex;
                continue;
            }


            mapInitialized = true;
            
            // --------------------------------------------------------
            // La deuxième image du bootstrap devient notre première
            // keyframe.
            // 
            // Sa pose monde -> caméra est exactement la pose relative
            // obtenue depuis la première caméra, puisque cette 
            // dernière définit le repère monde.
            // --------------------------------------------------------
            keyframeFrameIndex = frameIndex;
            keyframeKeypoints = currentKeypoints;
            keyframeDescriptors = currentDescriptors.clone();
            keyframeMapPointIndices = bootstrapMapPointIndices;
            keyframeWorldToCameraRotation = relativeRotation.clone();
            keyframeTranslationVector = relativeTranslation.clone();

            std::cout << "\n[KEYFRAME] Initial keyframe: frame "
                      << keyframeFrameIndex
                      << '\n';

            std::cout << "\n=== Map initialization ===\n";

            std::cout << "Bootstrap frame: "
                      << frameIndex
                      << '\n';

            std::cout << "Pose inliers: "
                      << poseInliers
                      << '\n';

            std::cout << "Initial 3D map points: "
                      << mapPoints.size()
                      << '\n';

            std::cout << "Map descriptors: "
                      << mapDescriptors.rows
                      << " x "
                      << mapDescriptors.cols
                      << "\n\n";
        }

        // -------------------------------------------------------------
        // Composition de la trajectoire
        //
        // recoverPose donne :
        //
        // X_current = R * X_previous + t
        //
        // Le centre de la caméra courante exprimé dans le repère
        // précédent vaut :
        //
        // C = -R^T t
        //
        // On le transforme ensuite dans notre repère monde.
        // -------------------------------------------------------------
        const cv::Mat relativeCameraCenter = 
            -relativeRotation.t() * relativeTranslation;

        cameraPositionWorld +=
            worldFromCameraRotation * relativeCameraCenter;

        worldFromCameraRotation =
            worldFromCameraRotation * relativeRotation.t();

        lastAcceptedPoseFrameIndex =
            frameIndex;

        ++successfulPoseSteps;


        // -------------------------------------------------------------
        // Sauvegarde de la trajectoire
        // -------------------------------------------------------------
        const double x =
            cameraPositionWorld.at<double>(0, 0);

        const double y =
            cameraPositionWorld.at<double>(1, 0);

        const double z =
            cameraPositionWorld.at<double>(2, 0);

        trajectoryFile << frameIndex
                       << ","
                       << static_cast<double>(frameIndex) / fps
                       << ","
                       << x
                       << ","
                       << y
                       << "," 
                       << z
                       << ","
                       << goodMatches.size()
                       << ","
                       << poseInliers
                       << '\n';


        std::cout << "[OK] frame "
                  << frameIndex
                  << " | matches: "
                  << goodMatches.size()
                  << " | pose inliers: "
                  << poseInliers
                  << " | position: ["
                  << x
                  << ","
                  << y
                  << ","
                  << z
                  << "]\n";


        // -------------------------------------------------------------
        // Actualisation de la frame de référence (la frame courante 
        // devient la frame de référence)
        // -------------------------------------------------------------
        previousGray = gray.clone();
        previousKeypoints = currentKeypoints;
        previousDescriptors = currentDescriptors.clone();

        ++frameIndex;
    }

    video.release();
    trajectoryFile.close();



    // -------------------------------------------------------------
    // Résumé
    // -------------------------------------------------------------
    std::cout << "\n=== VO summary ===\n";

    std::cout << "Selected frames: "
              << selectedFrames
              << '\n';

    std::cout << "Successful pose steps: "
                << successfulPoseSteps
                << '\n';

    std::cout << "Trajectory saved to: "
                << outputCsvPath.string()
                << '\n';

    std::cout << "Last frame index reached: "
              << frameIndex
              << '\n';


    return 0;
}

