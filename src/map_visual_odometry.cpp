#include <opencv2/core/utils/logger.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <array>

#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace fs = std::filesystem;

// ----------------------------------------------------------------
// Informations de qualité associées à un point 3D de la carte.
// 
// Cette structure reste parallèle à mapPoints:
//      mapPoints[i], mapDescriptors.row(i) et mapPointQualities[i]
//      représentent toujours le même point physique.
// ----------------------------------------------------------------
struct MapPointQuality
{
    // Frame à laquelle le point 3D a été triangulé
    int creationFrameIndex = -1;

    // Dernière frame dans laquelle ce point a été validé
    // comme inlier PnP
    // À sa création, on considère qu'il vient d'être vu.
    int lastSeenFrameIndex = -1;

    // Nombre de fois où le descripteur de ce point a produit une correspondance
    // 3D-2D candidate pour PnP
    int pnpMatchCount = 0;

    // Nombre de fois où cette correspondance a réellement survécu au RANSAC PnP
    int pnpInlierCount = 0;

    // Dernière frame dans laquelle le descripteur du point
    // a produit une correspondance 3D-2D après Lowe + déduplication 2D.
    // 
    // ATTENTION: cela ne signifie PAS que le point était un inlier PnP.
    int lastMatchedFrameIndex = -1;
};

using MapPointId = std::uint64_t;

// valeur réservée signifiant "aucun point 3D valide"
constexpr MapPointId invalidMapPointId =    
    std::numeric_limits<MapPointId>::max();

// -----------------------------------------------------------------
// Observation d'un point 3D dans une keyframe.
// 
// mapPointId est stable même si le culling compacte mapPoints.
// -----------------------------------------------------------------
struct KeyframeObservation
{
    MapPointId mapPointId = invalidMapPointId;

    cv::Point2d imagePoint;
};

// ----------------------------------------------------------------
// Infos nécessaires au futur Bundle Adjustment
// 
// Convention: X_camera = Rcw * X_world + tcw
// ----------------------------------------------------------------
struct KeyframeRecord
{
    int frameIndex = -1;

    cv::Mat worldToCameraRotation;
    cv::Mat translationVector;

    std::vector<KeyframeObservation> observations;
};


// --------------------------------------------------------------------------
// Pose locale utilisée par le Bundle Adjustment
// 
// parameters[0..2] : rotation angle-axis
// parameters[3..5] : translation tcw
// 
// Convention: X_camera = Rcw * X_world + tcw
// --------------------------------------------------------------------------
struct LocalBundleAdjustmentPose
{
    int frameIndex = -1;

    std::array<double, 6> parameters{};


    // Les deux plus anciennes poses seront fixées pour supprimer
    // les libertés de jauge du problème monoculaire.
    bool fixed = false;
};


// --------------------------------------------------------------------------
// Point 3D actif inclus dans la fenêtre locale
// --------------------------------------------------------------------------
struct LocalBundleAdjustmentPoint
{
    MapPointId mapPointId = invalidMapPointId;

    // Indice actuel dans mapPoints
    // 
    // Valable que pour l'état courant de la carte
    int globalMapPointIndex = -1;

    std::array<double, 3> position{};
};


// ----------------------------------------------------------------------
// Mesure 2D reliant une pose locale à un point 3D local
// ----------------------------------------------------------------------
struct LocalBundleAdjustmentObservation
{
    int poseIndex= -1;
    int pointIndex = -1;

    cv::Point2d imagePoint;
};


// ------------------------------------------------------------------------
// Problème complet envoyé plus tard à l'optimiseur
// ------------------------------------------------------------------------
struct LocalBundleAdjustmentProblem
{
    std::vector<LocalBundleAdjustmentPose> poses;
    std::vector<LocalBundleAdjustmentPoint> points;
    std::vector<LocalBundleAdjustmentObservation> observations;
};

// ------------------------------------------------------------------
// Résidu de reprojection utilisé par Ceres pour le Bundle
// Adjustment local.
// 
// Paramètres caméra:
//      camera[0..2] = rotation angle-axis
//      camera[3..5] = translation tcw
// 
// Paramètres point:
//      point[0..2] = coordonnées 3D dans le repère monde
// 
// Convention: X_camera = Rcw * X_world + tcw
// 
// Les observations 2D fournies à ce résidu sont déjà UNDISTORTED
// On utilise donc un modèle pinhole simple, sans k1, k2, p1, p2 ...
// ------------------------------------------------------------------
struct BundleAdjustmentReprojectionResidual
{
    BundleAdjustmentReprojectionResidual(
        double observedX, double observedY,
        double fx, double fy,
        double cx, double cy) :
        observedX(observedX), observedY(observedY),
        fx(fx), fy(fy), cx(cx), cy(cy)
    {
    }

    template <typename T> bool operator()(
        const T* const camera,
        const T* const point,
        T* residuals) const
    {
        // -----------------------------------------------------------
        // Rotation du point monde vers le repère caméra.
        // 
        // camera[0..2] est une rotation angle-axis.
        // -----------------------------------------------------------
        T pointCamera[3];

        ceres::AngleAxisRotatePoint(
            camera, point, pointCamera
        );


        // ------------------------------------------------------------
        // Translation monde -> caméra
        // ------------------------------------------------------------
        pointCamera[0] += camera[3];
        pointCamera[1] += camera[4];
        pointCamera[2] += camera[5];


        // -----------------------------------------------------------
        // Projection perspective pinhole
        // 
        // xn = Xc/Zc & yn = Yc/Zc
        // -----------------------------------------------------------
        const T normalizedX =
            pointCamera[0] / pointCamera[2];

        const T normalizedY =
            pointCamera[1] / pointCamera[2];

        // -----------------------------------------------------------
        // Passage aux coordonnées pixels
        // -----------------------------------------------------------
        const T projectedX =
            T(fx) * normalizedX + T(cx);

        const T projectedY =
            T(fy) * normalizedY + T(cy);


        // -----------------------------------------------------------
        // Résidu = projection prédite - observation mésurée
        // 
        // Ceres cherchera à rendre ces deux valeurs aussi proches de 
        // zéro que possible.
        // -----------------------------------------------------------
        residuals[0] =
            projectedX - T(observedX);

        residuals[1] =
            projectedY - T(observedY);

        return true;
    }

    double observedX = 0.0;
    double observedY = 0.0;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};


int cullOldNeverConfirmedMapPoints(
    int currentFrameIndex,
    int minimumPointAgeFrames,
    int maximumUnmatchedAgeFrames,

    std::vector<cv::Point3f>& mapPoints,
    cv::Mat& mapDescriptors,
    std::vector<MapPointQuality>& mapPointQualities,
    std::vector<MapPointId>& mapPointIds,

    std::vector<int>& keyframeMapPointIndices,
    std::vector<int>& currentMapPointIndices)
{
    // ----------------------------------------------------
    // Sécurité: les 3 structures de carte doivent être 
    // prafaitement synchronisées avant toute suppression.
    // ----------------------------------------------------
    if (mapPoints.size() !=
            static_cast<std::size_t>(mapDescriptors.rows) ||
        mapPoints.size() !=
            mapPointQualities.size() ||
        mapPoints.size() !=
            mapPointIds.size()
        )
    {
        std::cerr << "Error: inconsistant map before culling.\n";
        return -1;
    }


    const std::size_t oldMapSize = mapPoints.size();

    // oldToNew[i] = 
    //      nouvel indice du point i s'il est conservé,
    //      -1 s'il est supprimé
    std::vector<int> oldToNew(
        oldMapSize, -1
    );

    std::vector<cv::Point3f> newMapPoints;
    std::vector<MapPointQuality> newMapPointQualities;
    cv::Mat newMapDescriptors;
    std::vector<MapPointId> newMapPointIds;

    newMapPoints.reserve(oldMapSize);
    newMapPointQualities.reserve(oldMapSize);
    newMapPointIds.reserve(oldMapSize);

    int removedPoints = 0;
    int protectedByRecentMatch = 0;

    for (std::size_t oldIndex = 0; oldIndex < oldMapSize; ++oldIndex)
    {
        const MapPointQuality& quality = mapPointQualities[oldIndex];

        const int pointAgeFrames = currentFrameIndex - quality.creationFrameIndex;

        const bool oldEnough =
            pointAgeFrames >= minimumPointAgeFrames;

        const bool neverConfirmed =
            quality.pnpInlierCount == 0;

        // Le point est considéré récemment retrouvé s'il a produit 
        // un bon match dans la fenêtre temporelle récente.
        const bool recentlyMatched =
            quality.lastMatchedFrameIndex >= 0 &&
            (currentFrameIndex - quality.lastMatchedFrameIndex) < maximumUnmatchedAgeFrames;

        const bool shouldCull =
            oldEnough &&
            neverConfirmed &&
            !recentlyMatched;

        if (oldEnough && neverConfirmed && recentlyMatched)
        {
            ++protectedByRecentMatch;
        }

        if (shouldCull)
        {
            ++removedPoints;
            continue;
        }

        const int newIndex =
            static_cast<int>(
                newMapPoints.size()
            );

        oldToNew[oldIndex] = newIndex;

        newMapPoints.push_back(
            mapPoints[oldIndex]
        );

        newMapPointQualities.push_back(quality);

        newMapDescriptors.push_back(
            mapDescriptors.row(
                static_cast<int>(oldIndex)
            )
        );

        newMapPointIds.push_back(
            mapPointIds[oldIndex]
        );
    }

    // -------------------------------------------------------------
    // Les associations keypoint -> mapPoint utilisent les anciens
    // indices. Il faut donc impérativement les remapper.
    // -------------------------------------------------------------
    auto remapAssociations =
        [&oldToNew](std::vector<int>& associations)
    {
            for (int& mapPointIndex : associations)
            {
                if (mapPointIndex < 0)
                {
                    continue;
                }

                if (mapPointIndex >=
                    static_cast<int>(oldToNew.size()))
                {
                    mapPointIndex = -1;
                    continue;
                }

                mapPointIndex = oldToNew[mapPointIndex];
            }
    };

    remapAssociations(
        keyframeMapPointIndices
    );

    remapAssociations(
        currentMapPointIndices
    );


    // Installation de la carte compactée
    mapPoints.swap(
        newMapPoints
    );

    mapPointQualities.swap(
        newMapPointQualities
    );

    mapPointIds.swap(
        newMapPointIds
    );

    mapDescriptors = newMapDescriptors;

    std::cout << "[MAP CULLING]"
              << " removed: "
              << removedPoints
              << " | protected by recent match: "
              << protectedByRecentMatch
              << " | size: "
              << oldMapSize
              << " -> "
              << mapPoints.size()
              << '\n';

    return removedPoints;
}


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


bool createKeyframeRecord(
    int frameIndex,

    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<int>& keypointMapPointIndices,

    const std::vector<MapPointId>& mapPointIds,

    const cv::Mat& worldToCameraRotation,
    const cv::Mat& translationVector,

    KeyframeRecord& record)
{
    // Une association est définie pour chacun des keypoints.
    if (keypoints.size() != keypointMapPointIndices.size())
    {
        std::cerr << "Error: keyframe keypoint/association mismatch.\n";
        return false;
    }

    record = KeyframeRecord{};

    record.frameIndex = frameIndex;

    record.worldToCameraRotation = 
        worldToCameraRotation.clone();

    record.translationVector = 
        translationVector.clone();

    record.observations.reserve(
        keypoints.size()
    );


    for (std::size_t keypointIndex = 0;
        keypointIndex < keypoints.size();
        ++keypointIndex)
    {
        const int mapPointIndex = 
            keypointMapPointIndices[keypointIndex];

        // Ce keypoint n'a aucune association 3D.
        if (mapPointIndex < 0)
        {
            continue;
        }

        if (mapPointIndex >=
            static_cast<int>(mapPointIds.size())
            )
        {
            std::cerr << "Error: invalid map point index "
                      << "while rerecording keyframe"
                      << " | frame: "
                      << frameIndex
                      << " | map index: "
                      << mapPointIndex
                      << " | map size: "
                      << mapPointIds.size()
                      << '\n';

            return false;
        }


        KeyframeObservation observation;

        observation.mapPointId = 
            mapPointIds[mapPointIndex];

        observation.imagePoint = 
            cv::Point2d(
                static_cast<double>(
                    keypoints[keypointIndex].pt.x
                ),
                static_cast<double>(
                    keypoints[keypointIndex].pt.y
                )
            );

        record.observations.push_back(
            observation
        );
    }

    return true;
}

bool refreshStoredKeyframeRecord(
    std::vector<KeyframeRecord>& keyframeHistory,

    int frameIndex,

    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<int>& keypointMapPointIndices,

    const std::vector<MapPointId>& mapPointIds,

    const cv::Mat& worldToCameraRotation,
    const cv::Mat& translationVector)
{   
    auto keyframeIterator =
        std::find_if(
            keyframeHistory.rbegin(),
            keyframeHistory.rend(),

            [frameIndex](const KeyframeRecord& record)
            {
                return record.frameIndex == frameIndex;
            }
        );

    if (keyframeIterator == keyframeHistory.rend())
    {
        std::cerr << "Error: mapping anchor frame "
                  << frameIndex
                  << " not found in keyframe history.\n";

        return false;
    }


    KeyframeRecord refreshedRecord;

    if (!createKeyframeRecord(
        frameIndex,

        keypoints,
        keypointMapPointIndices,

        mapPointIds,

        worldToCameraRotation,
        translationVector,

        refreshedRecord))
    {
        return false;
    }

    std::cout << "[KEYFRAME REFRESH] frame: "
              << frameIndex
              << " | refreshed observations: "
              << refreshedRecord.observations.size()
              << '\n';

    *keyframeIterator = std::move(refreshedRecord);

    return true;
}

void printLocalBundleAdjustmentConnectivity(
    const std::vector<KeyframeRecord>& keyframeHistory,
    const std::vector<MapPointId>& activeMapPointIds,
    std::size_t windowSize)
{   
    if (keyframeHistory.empty())
    {
        return;
    }

    const std::size_t actualWindowSize = 
        std::min(
            windowSize,
            keyframeHistory.size()
        );

    const std::size_t firstKeyframeIndex = 
        keyframeHistory.size() - actualWindowSize;

    std::unordered_set<MapPointId> activeMapPointIdSet(
            activeMapPointIds.begin(),
            activeMapPointIds.end()
        );


    // Nombre de keyframes de la fenêtre dans lesquelles chaque point 3D est observé
    std::unordered_map<MapPointId, int> observationCounts;

    for (std::size_t keyframeIndex = firstKeyframeIndex;
        keyframeIndex < keyframeHistory.size();
        ++keyframeIndex)
    {
        const KeyframeRecord& keyframe =
            keyframeHistory[keyframeIndex];

        for (const KeyframeObservation& observation : keyframe.observations)
        {
            // Une ancienne keyframe peut encore contenir l'ID d'un point qui
            // a depuis été supprimé de la carte.
            // 
            // Ces observations historiques ne peuvent pas participer au Bundle 
            // Adjustment de la carte active.
            if (!activeMapPointIdSet.contains(observation.mapPointId))
            {
                continue;
            }
            
            ++observationCounts[
                observation.mapPointId
            ];
        }
    }

    int observedAtLeastOnce = 0;
    int observedAtLeastTwice = 0;
    int observedAtLeastThreeTimes = 0;
    int observedAtLeastFourTimes = 0;
    int observedInEveryKeyframe = 0;

    for (const auto& [mapPointId, count] : observationCounts)
    {
        (void)mapPointId;

        if (count >= 1) ++observedAtLeastOnce;
        if (count >= 2) ++observedAtLeastTwice;
        if (count >= 3) ++observedAtLeastThreeTimes;
        if (count >= 4) ++observedAtLeastFourTimes;
        if (count >=
            static_cast<int>(actualWindowSize))
        {
            ++observedInEveryKeyframe;
        }
    }

    std::cout << "[LOCAL BA WINDOW] keyframes: "
              << actualWindowSize
              << " | frames: ";

    for (std::size_t keyframeIndex = firstKeyframeIndex;
        keyframeIndex < keyframeHistory.size();
        ++keyframeIndex)
    {
        std::cout << keyframeHistory[keyframeIndex].frameIndex;

        if (keyframeIndex + 1 < keyframeHistory.size())
        {
            std::cout << ",";
        }
    }

    std::cout << " | unique points: "
              << observedAtLeastOnce
              << " | seen >= 2: "
              << observedAtLeastTwice
              << " | seen >= 3: "
              << observedAtLeastThreeTimes
              << " | seen >= 4: "
              << observedAtLeastFourTimes
              << " | seen in all: "
              << observedInEveryKeyframe
              << '\n';
}


bool buildLocalBundleAdjustmentProblem(
    const std::vector<KeyframeRecord>& keyframeHistory,

    const std::vector<cv::Point3f>& mapPoints,
    const std::vector<MapPointId>& mapPointIds,

    const cv::Mat& cameraMatrix,
    const cv::Mat& distortionCoefficients,

    std::size_t windowSize,

    LocalBundleAdjustmentProblem& problem)
{
    problem = LocalBundleAdjustmentProblem{};

    // ---------------------------------------------------------------
    // Vérif de cohérence de la carte active
    // ---------------------------------------------------------------
    if (mapPoints.size() != mapPointIds.size())
    {
        std::cerr << "Error: point/ID mismatch while building local BA\n";
        return false;
    }

    if (keyframeHistory.size() < windowSize)
    {
        return true;
    }

    const std::size_t firstKeyframeIndex = 
        keyframeHistory.size() - windowSize;

    // ----------------------------------------------------------------
    // ID stable -> indice courant dans mapPoints
    // 
    // Ceci permet de retrouver la position 3D actuelle d'un MapPointId
    // historique
    // ----------------------------------------------------------------
    std::unordered_map<MapPointId, int> activeMapPointIndices;

    activeMapPointIndices.reserve(
        mapPointIds.size()
    );

    for (std::size_t mapPointIndex = 0;
        mapPointIndex < mapPointIds.size();
        ++mapPointIndex)
    {
        const MapPointId mapPointId = mapPointIds[mapPointIndex];

        const auto [iterator, inserted] = 
            activeMapPointIndices.emplace(
                mapPointId,
                static_cast<int>(mapPointIndex)
            );

        (void)iterator;

        if (!inserted) 
        {
            std::cerr << "Error: duplicate MapPointId "
                      << mapPointId
                      << " in active map.\n";

            return false;
        }
    }


    // ---------------------------------------------------------------------
    // Première passe:
    // 
    // combien de keyframes différentes observent chaque point ?
    // 
    // Un point vu dans une seule keyframe n'est pas retenu pour le BA local
    // ---------------------------------------------------------------------
    std::unordered_map<MapPointId, int> observationCounts;

    for (std::size_t historyIndex = firstKeyframeIndex;
        historyIndex < keyframeHistory.size();
        ++historyIndex)
    {
        const KeyframeRecord& keyframe = 
            keyframeHistory[historyIndex];


        // Protection supplémentaire contre une éventuelle observation
        // dupliquée dans une même keyframe
        std::unordered_set<MapPointId> seenInThisKeyframe;

        for (const KeyframeObservation& observation : keyframe.observations)
        {
            if (!activeMapPointIndices.contains(
                observation.mapPointId
            ))
            {
                continue;
            }

            if (!seenInThisKeyframe.insert(
                observation.mapPointId
            ).second)
            {
                continue;
            }

            ++observationCounts[observation.mapPointId];
        }
    }


    // ------------------------------------------------------------------
    // Construction des 5 poses locales
    // ------------------------------------------------------------------
    problem.poses.reserve(
        windowSize
    );

    for (std::size_t localPoseIndex = 0;
        localPoseIndex < windowSize;
        ++localPoseIndex)
    {
        const KeyframeRecord& keyframe = 
            keyframeHistory[
                firstKeyframeIndex + localPoseIndex
            ];

        cv::Mat rotation64;
        cv::Mat translation64;

        keyframe.worldToCameraRotation.convertTo(
            rotation64, CV_64F
        );

        keyframe.translationVector.convertTo(
            translation64, CV_64F
        );


        // Conversion matrice de rotation -> vecteur angle-axis.
        // 
        // OpenCV appelle cette représentation un "rotation vector"
        cv::Mat rotationVector;

        cv::Rodrigues(
            rotation64, rotationVector
        );

        LocalBundleAdjustmentPose pose;

        pose.frameIndex = keyframe.frameIndex;

        pose.parameters[0] = rotationVector.at<double>(0, 0);

        pose.parameters[1] = rotationVector.at<double>(1, 0);

        pose.parameters[2] = rotationVector.at<double>(2, 0);

        pose.parameters[3] = translation64.at<double>(0, 0);

        pose.parameters[4] = translation64.at<double>(1, 0);

        pose.parameters[5] = translation64.at<double>(2, 0);


        // Deux poses fixes:
        // Cela va préserver le repère & l'échelle déjà établis par notre V0 monoculaire
        pose.fixed = localPoseIndex < 2;

        problem.poses.push_back(pose);
    }

    // --------------------------------------------------------------------------
    // Construction des points ayant au moins 2 observations.
    // 
    // On parcourt mapPointIds plutôt que l'unordered_map pour conserver un ordre
    // deterministe entre les exécutions.
    // --------------------------------------------------------------------------
    std::unordered_map<MapPointId, int> localPointIndices;

    for (std::size_t globalPointIndex = 0;
        globalPointIndex < mapPointIds.size();
        ++globalPointIndex)
    {
        const MapPointId mapPointId = mapPointIds[globalPointIndex];
        const auto countIterator = 
            observationCounts.find(mapPointId);

        if (countIterator == observationCounts.end())
        {
            continue;
        }

        if (countIterator->second < 2)
        {
            continue;
        }

        LocalBundleAdjustmentPoint point;

        point.mapPointId = mapPointId;
        point.globalMapPointIndex = 
            static_cast<int>(globalPointIndex);

        point.position[0] = 
            static_cast<double>(
                mapPoints[globalPointIndex].x
            );

        point.position[1] =
            static_cast<double>(
                mapPoints[globalPointIndex].y
            );

        point.position[2] =
            static_cast<double>(
                mapPoints[globalPointIndex].z
            );

        const int localPointIndex =
            static_cast<int>(
                problem.points.size()
            );

        localPointIndices.emplace(
            mapPointId, localPointIndex
        );

        problem.points.push_back(
            point
        );
    }


    // ---------------------------------------------------------------------
    // Deuxième passe: construction des arêtes 2D du graphe BA
    // ---------------------------------------------------------------------
    for (std::size_t localPoseIndex = 0;
        localPoseIndex < windowSize;
        ++localPoseIndex)
    {
        const KeyframeRecord& keyframe = 
            keyframeHistory[
                firstKeyframeIndex + localPoseIndex
            ];

        std::unordered_set<MapPointId> seenInThisKeyframe;

        for (const KeyframeObservation& observation : keyframe.observations)
        {
            const auto pointIterator = 
                localPointIndices.find(observation.mapPointId);

            if (pointIterator == localPointIndices.end())
            {
                continue;
            }

            if (!seenInThisKeyframe.insert(observation.mapPointId).second)
            {
                continue;
            }

            LocalBundleAdjustmentObservation localObservation;

            localObservation.poseIndex = 
                static_cast<int>(localPoseIndex);

            localObservation.pointIndex =
                pointIterator->second;

            std::vector<cv::Point2f> distortedPoints{
                cv::Point2f(
                    static_cast<float>(observation.imagePoint.x),
                    static_cast<float>(observation.imagePoint.y)
                )
            };

            std::vector<cv::Point2f> undistortedPoints;

            cv::undistortPoints(
                distortedPoints,
                undistortedPoints,
                cameraMatrix,
                distortionCoefficients,
                cv::noArray(),
                cameraMatrix
            );

            if (undistortedPoints.size() != 1)
            {
                std::cerr << "Error: failed to undistort BA observation.\n";
                return false;
            }


            localObservation.imagePoint = 
                cv::Point2d(
                    static_cast<double>(undistortedPoints[0].x),
                    static_cast<double>(undistortedPoints[0].y)
                );


            problem.observations.push_back(localObservation);
        }
    }


    // -------------------------------------------------------------------
    // Diagnostic par pose
    // -------------------------------------------------------------------
    std::vector<int> observationsPerPose(
        problem.poses.size(), 0
    );

    for (const LocalBundleAdjustmentObservation& observation : problem.observations)
    {
        ++observationsPerPose[observation.poseIndex];
    }

    std::cout << "[LOCAL BA PROBLEM] poses: "
              << problem.poses.size()
              << " | points: "
              << problem.points.size()
              << " | observations: "
              << problem.observations.size()
              << " | fixed poses: 2"
              << " | observations/pose: ";

    for (std::size_t poseIndex = 0;
        poseIndex < observationsPerPose.size();
        ++poseIndex)
    {
        std::cout << problem.poses[poseIndex].frameIndex
                  << ":"
                  << observationsPerPose[poseIndex];

        if (poseIndex + 1 < observationsPerPose.size())
        {
            std::cout << ",";
        }
    }

    std::cout << '\n';

    return true;
}

bool runLocalBundleAdjustment(
    LocalBundleAdjustmentProblem& localProblem,
    const cv::Mat& cameraMatrix)
{
    if (localProblem.poses.empty() ||
        localProblem.points.empty() ||
        localProblem.observations.empty())
    {
        std::cerr << "Error: empty local BA problem.\n";
        return false;
    }

    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        std::cerr << "Error: invalid camera matrix for local BA.\n";
        return false;
    }

    // ------------------------------------------------------------
    // Intrinsèques de la caméra
    // ------------------------------------------------------------
    cv::Mat cameraMatrix64;

    cameraMatrix.convertTo(
        cameraMatrix64, CV_64F
    );

    const double fx = 
        cameraMatrix64.at<double>(0, 0);

    const double fy =
        cameraMatrix64.at<double>(1, 1);

    const double cx =
        cameraMatrix64.at<double>(0, 2);

    const double cy =
        cameraMatrix64.at<double>(1, 2);


    // -------------------------------------------------------------
    // Création du problème Ceres
    // -------------------------------------------------------------
    ceres::Problem ceresProblem;


    // -------------------------------------------------------------
    // Ajout explicite des poses
    // 
    // Chaque pose = 6 params:
    //      angle-axis (3) & translation (3)
    // -------------------------------------------------------------
    for (LocalBundleAdjustmentPose& pose : localProblem.poses)
    {
        ceresProblem.AddParameterBlock(
            pose.parameters.data(), 6
        );

        // Fixation des deux plus anciennes poses
        if (pose.fixed)
        {
            ceresProblem.SetParameterBlockConstant(
                pose.parameters.data()
            );
        }
    }

    // --------------------------------------------------------------
    // Points 3D
    // --------------------------------------------------------------
    for (LocalBundleAdjustmentPoint& point : localProblem.points)
    {
        ceresProblem.AddParameterBlock(
            point.position.data(), 3
        );
    }

    // --------------------------------------------------------------
    // Huber loss
    // 
    // Une observation ayant une grosse erreur de reprojection aura 
    // moins d'influence sur l'optimisation.
    // 
    // Delta = 2 px
    // --------------------------------------------------------------
    ceres::LossFunction* huberLoss =
        new ceres::HuberLoss(2.0);


    // --------------------------------------------------------------
    // Une observation 2D = un bloc résiduel de dimension 2
    // --------------------------------------------------------------
    for (const LocalBundleAdjustmentObservation& observation : 
            localProblem.observations)
    {
        if (observation.poseIndex < 0 ||
            observation.poseIndex >=
            static_cast<int>(localProblem.poses.size()) ||
            observation.pointIndex < 0 ||
            observation.pointIndex >=
            static_cast<int>(localProblem.points.size()))
        {
            std::cerr << "Error: invalid local BA observation index.\n";
            return false;
        }

        LocalBundleAdjustmentPose& pose =
            localProblem.poses[observation.poseIndex];

        LocalBundleAdjustmentPoint& point =
            localProblem.points[observation.pointIndex];

        ceres::CostFunction* costFunction =
            new ceres::AutoDiffCostFunction<
                BundleAdjustmentReprojectionResidual, 2, 6, 3
            >(
                new BundleAdjustmentReprojectionResidual(
                    observation.imagePoint.x, observation.imagePoint.y,
                    fx, fy,
                    cx, cy
                )
            );

        ceresProblem.AddResidualBlock(
            costFunction,
            huberLoss,
            pose.parameters.data(),
            point.position.data()
        );
    }

    // ---------------------------------------------------------------
    // Mesure AVANT optimisation
    // 
    // Ici, on désactive volontairement la Huber Loss afin de calculer
    // une vraie erreur quadratique de reprojection.
    // 
    // Ceres définit: cost = 1/2 * somme(résidu²)
    // Donc RMSE_2D = sqrt(2 cost/nbre_observations)
    // ---------------------------------------------------------------
    ceres::Problem::EvaluateOptions evaluateOptions;

    evaluateOptions.apply_loss_function = false;

    double initialRawCost = 0.0;

    if (!ceresProblem.Evaluate(
        evaluateOptions,
        &initialRawCost,
        nullptr,
        nullptr,
        nullptr))
    {
        std::cerr << "Error: could not evaluate local BA before solving.\n";
        return false;
    }


    const double initialRmse =
        std::sqrt(
            2.0 * initialRawCost /
            static_cast<double>(localProblem.observations.size())
        );


    // ---------------------------------------------------------------
    // Configuration du solveur
    // 
    // Il n'y a que 5 poses dans notre fenêtre.
    // DENSE_SCHUR est donc très adapté et simple pour ce premier BA.
    // ---------------------------------------------------------------
    ceres::Solver::Options options;

    options.linear_solver_type =
        ceres::DENSE_SCHUR;

    options.max_num_iterations = 30;

    options.num_threads = 1;

    // ---------------------------------------------------------------
    // Optimisation
    // ---------------------------------------------------------------
    ceres::Solver::Summary summary;

    ceres::Solve(
        options, &ceresProblem, &summary
    );

    // ---------------------------------------------------------------
    // Mesure APRES optimisation
    // ---------------------------------------------------------------
    double finalRawCost = 0.0;

    if (!ceresProblem.Evaluate(
        evaluateOptions, &finalRawCost,
        nullptr, nullptr, nullptr))
    {
        std::cerr << "Error: could not evaluate local BA after solving.\n";
        return false;
    }

    const double finalRmse =
        std::sqrt(
            2.0 * finalRawCost /
            static_cast<double>(localProblem.observations.size())
        );

    // ---------------------------------------------------------------
    // Diagnostic
    // ---------------------------------------------------------------
    std::cout << "[LOCAL BA] poses: "
        << localProblem.poses.size()
        << " | points: "
        << localProblem.points.size()
        << " | observations: "
        << localProblem.observations.size()
        << " | RMSE: "
        << initialRmse
        << " -> "
        << finalRmse
        << " px"
        << " | iterations: "
        << summary.iterations.size()
        << " | usable: "
        << (summary.IsSolutionUsable() ? "yes" : "no")
        << '\n';

    return summary.IsSolutionUsable();
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
    //const int minimumPnPInliers = 20;
    const int minimumPnPInliers = 15;

    const double minimumPnPInlierRatio = 0.40;

    // Entre 15 & 19 inliers, on exigera une proportion nettement plus
    // forte de correspondances cohérentes
    const double minimumSparsePnPInlierRatio = 0.60;


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

    std::vector<MapPointQuality> mapPointQualities;


    // Identifiant stable de chaque point 3D.
    // Contrairement à l'indice dans mapPoints, cet ID ne change
    // jamais lors d'un culling.
    std::vector<MapPointId> mapPointIds;

    MapPointId nextMapPointId = 0;

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
    
    // Dernière frame ayant déclenché la politique de sélection
    // de keyframe.
    // 
    // Elle est distincte de keyframeFrameIndex :
    // keyframeframeIndex désigne désormais l'ancre réellement 
    // utilisée pour la triangulation de nouveaux points 3D.
    int lastSelectedKeyframeFrameIndex = -1;

    // Une keyframe candidate ne devient nouvelle ancre de 
    // triangulation que si elle a permis d'enrichir suffisamment
    // la carte.

    // Une extension d'au moins 10 points est suffisamment utile pour
    // justifier une nouvelle ancre.
    const int minimumNewMapPointsForAnchorPromotion = 10;

    // Même si peu de nouveaux points ont été créés, on ne laisse pas
    // l'ancre devenir trop trop ancienne.
    // 
    // 4 * frameStep = environ 2 secondes
    const int maximumMappingAnchorGapFrames = 4 * frameStep;

    // Historique des keyframes sélectionnées.
    // 
    // Pour l'instant on les conserve toutes.
    // Le local BA utilisera ensuite uniquement les dernières
    // keyframes de cet historique.
    std::vector<KeyframeRecord> keyframeHistory;

    keyframeHistory.reserve(64);



    // ----------------------------------------------------------------
    // Politique adaptative de création des keyframes.
    // Minimum: 
    // On évite 2 keyframes trop proches, ce qui donnerait 
    // peu de parallaxe.
    // 
    // Maximum:
    // même si le tracking reste bon, on renouvelle périodiquement
    // la référence.
    // ----------------------------------------------------------------
    const int minimumKeyframeGapFrames = 2 * frameStep;  // environ 1 seconde
    const int maximumKeyframeGapFrames = 6 * frameStep;  // environ 3 seconde

    // Tracking considéré comme affaibli si l'un de ces critères devient mauvais
    const int keyframeInlierThreshold = 50;
    const double keyframeInlierRatioThreshold = 0.60;

    const std::size_t keyframeMatchThreshold = 60;
    
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


            // ---------------------------------------------------
            // Matches ayant passé le test de Lowe
            //
            // Ils ne sont pas encore nécessairement univoques 
            // côté image courante : plusieurs points 3D peuvent
            // avoir choisi le même keypoint 2D
            // ---------------------------------------------------
            std::vector<cv::DMatch> mapCandidateMatches;
            
            for (const auto& candidates : mapKnnMatches)
            {
                if (candidates.size() < 2)
                {
                    continue;
                }

                if (candidates[0].distance < ratioThreshold * candidates[1].distance)
                {
                    mapCandidateMatches.push_back(candidates[0]);
                }
            }

            std::sort(
                mapCandidateMatches.begin(),
                mapCandidateMatches.end(),
                [](const cv::DMatch& a,
                    const cv::DMatch& b)
                {
                    return a.distance < b.distance;
                }
            );

            std::vector<bool> currentKeypointAlreadyUsed(
                currentKeypoints.size(), false
            );

            std::vector<cv::DMatch> mapGoodMatches;

            mapGoodMatches.reserve(
                mapCandidateMatches.size()
            );

            for (const cv::DMatch& match : mapCandidateMatches)
            {
                // trainIdx correspond au keypoint de la frame courante
                if (currentKeypointAlreadyUsed[match.trainIdx])
                {
                    continue;
                }

                currentKeypointAlreadyUsed[match.trainIdx] = true;

                mapGoodMatches.push_back(
                    match
                );
            }

            const std::size_t removedDuplicateImageMatches =
                mapCandidateMatches.size() - mapGoodMatches.size();

            std::cout << "[MAP MATCHING] frame "
                      << frameIndex
                      << " | Lowe matches: "
                      << mapCandidateMatches.size()
                      << " | unique 2D matches: "
                      << mapGoodMatches.size()
                      << " | duplicates removed: "
                      << removedDuplicateImageMatches
                      << '\n';

            // ---------------------------------------------------------
            // Dernière apparition visuelle des points de carte.
            // 
            // mapGoodMatches a déjà passé:
            //      - Lowe
            //      - l'unicité côté keypoint 2D.
            // 
            // On met cette info à jour AVANT PnP.
            // Ainsi, même si la pose est ensuite rejetée, on sait que
            // le descripteur du point est encore retrouvé dans l'image.
            // ---------------------------------------------------------
            for (const cv::DMatch& match : mapGoodMatches)
            {
                const int mapPointIndex = match.queryIdx;

                if (mapPointIndex < 0 ||
                    mapPointIndex >= static_cast<int>(
                        mapPointQualities.size()
                        ))
                {
                    std::cerr << "Error: invalid map point index while updating last match.\n";
                    return 1;
                }

                mapPointQualities[mapPointIndex].lastMatchedFrameIndex = frameIndex;
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

            // Avec les correspondances 2D désormais univoques, 15 à 19 
            // inliers peuvent être suffisants si leur ratio est 
            // particulièrement bon
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

            // Cas "sparse but clean": 
            // entre 15 & 19 inliers, on demande au moins 60%
            if (pnpInliers.rows < 20 &&
                pnpInlierRatio < minimumSparsePnPInlierRatio)
            {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : sparse PnP"
                          << " | inliers: "
                          << pnpInliers.rows
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
            // Association des keypoints de la frame courante avec les 
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
            
            // Tous les matches envoyés à PnP comptent comme tentative
            // d'observation du point 3D
            for (const cv::DMatch& match : mapGoodMatches)
            {
                const int mapPointIndex = match.queryIdx;

                ++mapPointQualities[mapPointIndex].pnpMatchCount;
            }

            // Les inliers PnP sont des observations confirmées
            for (int i = 0; i < pnpInliers.rows; ++i)
            {
                const int matchListIndex = 
                    pnpInliers.at<int>(i, 0);

                const int mapPointIndex = 
                    mapGoodMatches[matchListIndex].queryIdx;

                MapPointQuality& quality = 
                    mapPointQualities[mapPointIndex];

                ++quality.pnpInlierCount;

                quality.lastSeenFrameIndex = frameIndex;
            }

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
            // Décision adaptative de création d'une nouvelle keyframe
            // -----------------------------------------------------------
            const int framesSinceLastKeyframe = 
                frameIndex - lastSelectedKeyframeFrameIndex;

            // Trop proche de la dernière keyframe:
            // on préfère attendre davantage de baseline
            const bool enoughFrameSeparation = 
                framesSinceLastKeyframe >= minimumKeyframeGapFrames;

            // Même avec un bon tracking, on ne conserve pas une keyframe indéfiniment
            const bool maximumGapReached = 
                framesSinceLastKeyframe >= maximumKeyframeGapFrames;

            // ---------------------------------------------------------
            // Qualité actuelle du tracking
            // ---------------------------------------------------------

            // Peu de correspondances carte -> image
            const bool lowMapVisibility =
                mapGoodMatches.size() < keyframeMatchThreshold;

            // Peu d'inliers géométriquement cohérents
            const bool lowAbsolutePnPSupport =
                pnpInliers.rows < keyframeInlierThreshold;

            // Une proportion importante des matches est rejetée par RANSAC
            const bool lowPnPInlierRatio =
                pnpInlierRatio < keyframeInlierRatioThreshold;

            const bool trackingIsWeakening =
                lowMapVisibility ||
                lowAbsolutePnPSupport ||
                lowPnPInlierRatio;

            // -------------------------------------------------------
            // Création de la keyframe
            // Possibilités:
            //      1. le tracking commence à faiblir ET suffisamment
            //          de temps s'est écoulé;
            //      2. le délai maximal est atteint, même si le 
            //          tracking reste encore bon. 
            // -------------------------------------------------------
            const bool shouldCreateKeyframe = 
                maximumGapReached || 
                (
                    enoughFrameSeparation && 
                    trackingIsWeakening
                );

            if (shouldCreateKeyframe)
            {
                std::cout << "[KEYFRAME] frame "
                          << frameIndex
                          << " selected"
                          << " | reason: "
                          << (
                                maximumGapReached ? "maximum gap" : "tracking weakening"
                              )
                          << " | gap: "
                          << framesSinceLastKeyframe
                          << " frames"
                          << " | matches: "
                          << mapGoodMatches.size()
                          << " | PnP inliers: "
                          << pnpInliers.rows
                          << " | ratio: "
                          << pnpInlierRatio
                          << " | previous selected: "
                          << lastSelectedKeyframeFrameIndex
                          << " | mapping anchor: "
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
                
                // -------------------------------------------------------
                // Création des infos de qualité pour les nouveaux points
                // 3D ajoutés pendant cette extension de carte.
                // 
                // Avant addNewMapPointsBetweenKeyframes():
                //      mapSizeBefore = ancien nombre de points.
                // 
                // Après:
                //      les indices [mapSizeBefore, mapPoints.size()]
                //      correspondent exactement aux nouveaux points.
                // -----------------------------------------------------
                for (std::size_t mapPointIndex = mapSizeBefore;
                    mapPointIndex < mapPoints.size();
                    ++mapPointIndex)
                {
                    MapPointQuality quality;

                    quality.creationFrameIndex = frameIndex;
                    quality.lastSeenFrameIndex = frameIndex;
                    quality.lastMatchedFrameIndex = frameIndex;

                    mapPointQualities.push_back(
                        quality
                    );

                    mapPointIds.push_back(
                        nextMapPointId++
                    );
                }

                // Verif permanente de notre invariant
                if (mapPoints.size() != 
                        static_cast<std::size_t>(mapDescriptors.rows) || 
                    mapPoints.size() != 
                        mapPointQualities.size() ||
                    mapPoints.size() != 
                        mapPointIds.size()
                   )
                {
                    std::cerr << "Error: map data size mismatch"
                              << " | points: "
                              << mapPoints.size()
                              << " | descriptors: "
                              << mapDescriptors.rows
                              << " | qualities: "
                              << mapPointQualities.size()
                              << " | IDs: "
                              << mapPointIds.size()
                              << '\n';

                    return 1;
                }


                // -----------------------------------------------------------
                // Diagnostic de qualité de la carte
                // Aucun point n'est supprimé pour l'instant
                // -----------------------------------------------------------
                int neverConfirmedByPnP = 0;
                int confirmedAtLeastOnce = 0;
                int confirmedAtLeastThreeTimes = 0;
                int staleMapPoints = 0;

                int oldNeverConfirmed = 0;
                int lowPnPConfirmationRatio = 0;

                const int stalePointAgeFrames =
                    12 * frameStep;                               // environ 6 secondes

                for (const MapPointQuality& quality : mapPointQualities)
                {
                    if (quality.pnpInlierCount == 0)
                    {
                        ++neverConfirmedByPnP;
                    }

                    if (quality.pnpInlierCount >= 1)
                    {
                        ++confirmedAtLeastOnce;
                    }

                    if (quality.pnpInlierCount >= 3)
                    {
                        ++confirmedAtLeastThreeTimes;
                    }

                    const int framesSinceLastSeen =
                        frameIndex - quality.lastSeenFrameIndex;

                    if (framesSinceLastSeen >= stalePointAgeFrames)
                    {
                        ++staleMapPoints;
                    }

                    const int pointAgeFrames =
                        frameIndex - quality.creationFrameIndex;

                    // Point suffisamment ancien pour avoir eu plusieurs occasions
                    // d'être réobservé, mais jamais validé par PnP.
                    if (pointAgeFrames >= 12 * frameStep &&
                        quality.pnpInlierCount == 0)
                    {
                        ++oldNeverConfirmed;
                    }

                    // Point qui a souvent produit un match, mais qui est rarement
                    // accepté comme inlier PnP.
                    // On exige au moins 5 tentatives pour ne pas tirer de 
                    // conclusion à partir de trop peu d'observations.
                    if (quality.pnpMatchCount >= 5)
                    {
                        const double confirmationRatio = 
                            static_cast<double>(quality.pnpInlierCount) /
                            static_cast<double>(quality.pnpMatchCount);

                        if (confirmationRatio < 0.25)
                        {
                            ++lowPnPConfirmationRatio;
                        }
                    }
                }

                std::cout << "[MAP QUALITY]"
                          << " size: "
                          << mapPoints.size()
                          << " | never PnP-confirmed: "
                          << neverConfirmedByPnP
                          << " | confirmed >= 1: "
                          << confirmedAtLeastOnce
                          << " | confirmed >= 3: "
                          << confirmedAtLeastThreeTimes
                          << " | stale: "
                          << staleMapPoints
                          << " | old never-confirmed: "
                          << oldNeverConfirmed
                          << " | low confirmation ratio: "
                          << lowPnPConfirmationRatio
                          << '\n';


                const int minimumCullingAgeFrames = 12 * frameStep;

                // Un point qui a encore produit un bon match au cours des
                // 4 derniers intervalles (~2 secondes) est protégé du culling
                const int maximumUnmatchedAgeFrames = 4 * frameStep;

                const int culledMapPoints = 
                    cullOldNeverConfirmedMapPoints(
                        frameIndex,
                        minimumCullingAgeFrames,
                        maximumUnmatchedAgeFrames,

                        mapPoints,
                        mapDescriptors,
                        mapPointQualities,
                        mapPointIds,

                        keyframeMapPointIndices,
                        currentMapPointIndices
                    );

                if (culledMapPoints < 0)
                {
                    return 1;
                }

                if (mapPoints.size() != static_cast<std::size_t>(mapDescriptors.rows) ||
                    mapPoints.size() != mapPointQualities.size() ||
                    mapPoints.size() != mapPointIds.size())
                {
                    std::cerr << "Error: map data size mismatch after culling"
                              << " | points: "
                              << mapPoints.size()
                              << " | descriptors: "
                              << mapDescriptors.rows
                              << " | qualities: "
                              << mapPointQualities.size()
                              << " | IDs: "
                              << mapPointIds.size()
                              << '\n';

                    return 1;
                }


                // ------------------------------------------------------------------
                // La triangulation a potentiellement ajouté de nouvelles
                // associations au mapping anchor.
                // 
                // Son ancien KeyframeRecord a été créé avant que ces nouveaux points
                // actuelles, après remapping du culling.
                // ------------------------------------------------------------------
                if (!refreshStoredKeyframeRecord(
                    keyframeHistory,

                    keyframeFrameIndex,

                    keyframeKeypoints,
                    keyframeMapPointIndices,

                    mapPointIds,

                    keyframeWorldToCameraRotation,
                    keyframeTranslationVector))
                {
                    return 1;
                }

                // ----------------------------------------------------------------
                // Sauvegarde de la keyframe dans l'hstorique destiné au 
                // Bundle Adjustement.
                // 
                // Le culling vient d'être effectué:
                // currentMapPointIndices utilise donc déjà les indices compactés
                // de la carte.
                // 
                // createKeyframeRecord() convertit immédiatement ces indices
                // temporaires en MapPointId permanents.
                // ----------------------------------------------------------------
                KeyframeRecord currentKeyframeRecord;

                if (!createKeyframeRecord(
                    frameIndex,

                    currentKeypoints,
                    currentMapPointIndices,

                    mapPointIds,

                    worldToCameraRotation,
                    translationVector,

                    currentKeyframeRecord))
                {
                    return 1;
                }

                keyframeHistory.push_back(
                    std::move(currentKeyframeRecord)
                );

                

                std::cout << "[KEYFRAME HISTORY] frame: "
                          << frameIndex
                          << " | observations: "
                          << keyframeHistory.back().observations.size()
                          << " | stored keyframes: "
                          << keyframeHistory.size()
                          << '\n';


                printLocalBundleAdjustmentConnectivity(
                    keyframeHistory,
                    mapPointIds,
                    5
                );


                if (keyframeHistory.size() >= 5)
                {
                    LocalBundleAdjustmentProblem localBundleAdjustmentProblem;

                    if (!buildLocalBundleAdjustmentProblem(
                        keyframeHistory,
                        mapPoints, mapPointIds,
                        cameraMatrix,distortionCoefficients,
                        5,
                        localBundleAdjustmentProblem))
                    {
                        return 1;
                    }

                    if (!runLocalBundleAdjustment(
                        localBundleAdjustmentProblem, cameraMatrix))
                    {
                        std::cerr << "Error: local bundle adjustment failed.\n";
                        return 1;
                    }
                }
                
                // -------------------------------------------------------------
                // Sauvegarde 
                // -------------------------------------------------------------
                
                // La frame courante devient maintenant la nouvelle keyframe
                lastSelectedKeyframeFrameIndex = frameIndex;


                const int framesSinceMappingAnchor = 
                    frameIndex - keyframeFrameIndex;

                const bool enoughNewMapPoints = 
                    addedMapPoints >= minimumNewMapPointsForAnchorPromotion;

                const bool mappingAnchorTooOld = 
                    framesSinceMappingAnchor >= maximumMappingAnchorGapFrames;


                // -----------------------------------------------------------
                // Promotion éventuelle de la frame courante comme nouvelle
                // ancre de triangulation.
                // 
                // La sélection de la keyframe et le changement d'ancre sont
                // désormais deux décisions différentes.
                // -----------------------------------------------------------
                if (enoughNewMapPoints || mappingAnchorTooOld)
                {
                    keyframeFrameIndex = frameIndex;

                    keyframeKeypoints = currentKeypoints;
                    keyframeDescriptors = currentDescriptors.clone();

                    keyframeMapPointIndices = currentMapPointIndices;

                    keyframeWorldToCameraRotation = worldToCameraRotation.clone();
                    keyframeTranslationVector = translationVector.clone();

                    std::cout << "[MAP ANCHOR] promoted frame: "
                              << frameIndex
                              << " | reason: "
                              << (
                                    enoughNewMapPoints ? "map growth" : "anchor age"
                                  )
                              << " | added points: "
                              << addedMapPoints
                              << " | previous anchor age: "
                              << framesSinceMappingAnchor
                              << " frames\n";
                }
                else
                {
                    std::cout << "[MAP ANCHOR] kept frame: "
                              << keyframeFrameIndex
                              << " | candidate frame: "
                              << frameIndex
                              << " | added only: "
                              << addedMapPoints
                              << " | anchor age: "
                              << framesSinceMappingAnchor
                              << " frames\n";
                }
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

            // -----------------------------------------------------------
            // Initialisation des informations de qualité des points créés
            // pendant le bootstrap
            // 
            // On ne compte pas encore leurs deux observations de 
            // triangulation comme observations PnP.
            // 
            // pnpMatchCount / pnpInlierCount commencent donc à 0.
            // -----------------------------------------------------------
            mapPointQualities.clear();

            mapPointQualities.resize(
                mapPoints.size()
            );

            for (MapPointQuality& quality : mapPointQualities)
            {
                quality.creationFrameIndex = frameIndex;
                quality.lastSeenFrameIndex = frameIndex;
                quality.lastMatchedFrameIndex = frameIndex;
            }

            mapPointIds.clear();

            mapPointIds.reserve(
                mapPoints.size()
            );

            for (std::size_t i = 0; i < mapPoints.size(); ++i)
            {
                mapPointIds.push_back(
                    nextMapPointId++
                );
            }


            if (mapPointQualities.size() != mapPoints.size() ||
                mapPointIds.size() != mapPoints.size())
            {
                std::cerr << "Error: map data mismatch after bootstrap"
                          << " | points: "
                          << mapPoints.size()
                          << " | qualities: "
                          << mapPointQualities.size()
                          << " | IDs: "
                          << mapPointIds.size()
                          << "\n";
                return 1;
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

            lastSelectedKeyframeFrameIndex = frameIndex;

            KeyframeRecord initialKeyframeRecord;

            if (!createKeyframeRecord(
                frameIndex,

                currentKeypoints,
                bootstrapMapPointIndices,

                mapPointIds,

                relativeRotation,
                relativeTranslation,

                initialKeyframeRecord))
            {
                return 1;
            }


            keyframeHistory.push_back(
                std::move(initialKeyframeRecord)
            );

            std::cout << "[KEYFRAME HISTORY]"
                      << " frame: "
                      << frameIndex
                      << " | observations: "
                      << keyframeHistory.back().observations.size()
                      << " | stored keyframes: "
                      << keyframeHistory.size()
                      << '\n';

            printLocalBundleAdjustmentConnectivity(
                keyframeHistory,
                mapPointIds,
                5
            );

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

