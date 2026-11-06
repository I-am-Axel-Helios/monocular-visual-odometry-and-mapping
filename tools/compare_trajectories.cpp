#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;


struct TrajectoryPoint {
    int frameIndex = 0;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};


bool loadTrajectory(
    const fs::path& csvPath,
    std::vector<TrajectoryPoint>& trajectory
) {
    std::ifstream file(csvPath);

    if (!file.is_open()) {
        std::cerr
            << "Error: could not open trajectory file: "
            << csvPath
            << '\n';

        return false;
    }

    std::string line;

    // Header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream lineStream(line);

        std::vector<std::string> fields;
        std::string field;

        while (std::getline(
            lineStream,
            field,
            ','
        )) {
            fields.push_back(field);
        }

        if (fields.size() < 5) {
            continue;
        }

        try {
            TrajectoryPoint point;

            point.frameIndex =
                std::stoi(fields[0]);

            point.x =
                std::stod(fields[2]);

            point.y =
                std::stod(fields[3]);

            point.z =
                std::stod(fields[4]);

            trajectory.push_back(point);
        }
        catch (...) {
            std::cerr
                << "Warning: invalid CSV line ignored:\n"
                << line
                << '\n';
        }
    }

    return trajectory.size() >= 2;
}


double computePathLength(
    const std::vector<TrajectoryPoint>& trajectory
) {
    double pathLength = 0.0;

    for (std::size_t i = 1;
        i < trajectory.size();
        ++i) {

        const double dx =
            trajectory[i].x -
            trajectory[i - 1].x;

        const double dy =
            trajectory[i].y -
            trajectory[i - 1].y;

        const double dz =
            trajectory[i].z -
            trajectory[i - 1].z;

        pathLength += std::sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        );
    }

    return pathLength;
}


double computeEndDisplacement(const std::vector<TrajectoryPoint>& trajectory) {
    const double dx = trajectory.back().x - trajectory.front().x;
    const double dy = trajectory.back().y - trajectory.front().y;
    const double dz = trajectory.back().z - trajectory.front().z;

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}


bool normalizeTrajectory(std::vector<TrajectoryPoint>&trajectory) {
    const double startX = trajectory.front().x;
    const double startY = trajectory.front().y;
    const double startZ = trajectory.front().z;

    const double endDisplacement = computeEndDisplacement(trajectory);

    if (!std::isfinite(endDisplacement) || endDisplacement < 1e-9) {
        return false;
    }

    for (TrajectoryPoint& point : trajectory) {
        point.x = (point.x - startX) / endDisplacement;
        point.y = (point.y - startY) / endDisplacement;
        point.z = (point.z - startZ) / endDisplacement;
    }

    return true;
}

bool buildCommonXZTrajectories(
    const std::vector<TrajectoryPoint>& pairwiseTrajectory,
    const std::vector<TrajectoryPoint>& mapBasedTrajectory,
    Eigen::MatrixXd& pairwiseCommonXZ,
    Eigen::MatrixXd& mapBasedCommonXZ,
    std::vector<int>& commonFrames) {

    std::vector<Eigen::Vector2d> pairwisePoints;
    std::vector<Eigen::Vector2d> mapBasedPoints;

    std::size_t pairwiseIndex = 0;
    std::size_t mapBasedIndex = 0;

    while (pairwiseIndex < pairwiseTrajectory.size() &&
        mapBasedIndex < mapBasedTrajectory.size()) {

        const int pairwiseFrame =
            pairwiseTrajectory[pairwiseIndex].frameIndex;

        const int mapBasedFrame =
            mapBasedTrajectory[mapBasedIndex].frameIndex;

        if (pairwiseFrame == mapBasedFrame) {
            pairwisePoints.emplace_back(
                pairwiseTrajectory[pairwiseIndex].x,
                pairwiseTrajectory[pairwiseIndex].z
            );

            mapBasedPoints.emplace_back(
                mapBasedTrajectory[mapBasedIndex].x,
                mapBasedTrajectory[mapBasedIndex].z
            );

            commonFrames.push_back(
                pairwiseFrame
            );

            ++pairwiseIndex;
            ++mapBasedIndex;
        }
        else if (pairwiseFrame < mapBasedFrame) {
            ++pairwiseIndex;
        }
        else {
            ++mapBasedIndex;
        }
    }

    if (commonFrames.size() < 3) {
        return false;
    }

    pairwiseCommonXZ.resize(
        2,
        commonFrames.size()
    );

    mapBasedCommonXZ.resize(
        2,
        commonFrames.size()
    );

    for (std::size_t i = 0;
        i < commonFrames.size();
        ++i) {

        pairwiseCommonXZ.col(i) =
            pairwisePoints[i];

        mapBasedCommonXZ.col(i) =
            mapBasedPoints[i];
    }

    return true;
}


double computeMean(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    double sum = 0.0;

    for (const double value : values) {
        sum += value;
    }

    return sum /
        static_cast<double>(values.size());
}


double computeMedian(
    std::vector<double> values
) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t middle =
        values.size() / 2;

    if (values.size() % 2 == 0) {
        return 0.5 *
            (
                values[middle - 1] +
                values[middle]
                );
    }

    return values[middle];
}


double computeRmse(
    const std::vector<double>& values
) {
    if (values.empty()) {
        return 0.0;
    }

    double squaredSum = 0.0;

    for (const double value : values) {
        squaredSum +=
            value * value;
    }

    return std::sqrt(
        squaredSum /
        static_cast<double>(values.size())
    );
}


int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_ERROR
    );

    if (argc < 4) {
        std::cerr
            << "Usage: compare_trajectories "
            << "<pairwise_csv> "
            << "<map_based_csv> "
            << "<output_image>\n";

        return 1;
    }

    const fs::path pairwiseCsvPath =
        argv[1];

    const fs::path mapBasedCsvPath =
        argv[2];

    const fs::path outputImagePath =
        argv[3];


    std::vector<TrajectoryPoint> pairwiseTrajectory;
    std::vector<TrajectoryPoint> mapBasedTrajectory;

    if (!loadTrajectory(
        pairwiseCsvPath,
        pairwiseTrajectory)) {

        return 1;
    }

    if (!loadTrajectory(
        mapBasedCsvPath,
        mapBasedTrajectory)) {

        return 1;
    }

    Eigen::MatrixXd pairwiseCommonXZ;
    Eigen::MatrixXd mapBasedCommonXZ;

    std::vector<int> commonFrames;

    if (!buildCommonXZTrajectories(
        pairwiseTrajectory,
        mapBasedTrajectory,
        pairwiseCommonXZ,
        mapBasedCommonXZ,
        commonFrames)) {

        std::cerr
            << "Error: not enough common trajectory frames.\n";

        return 1;
    }

    const Eigen::MatrixXd similarityTransform =
        Eigen::umeyama(
            pairwiseCommonXZ,
            mapBasedCommonXZ,
            true
        );

    // ------------------------------------------------------------
// Divergence entre les trajectoires sur les frames communes,
// après alignement du pairwise sur le map-based.
// ------------------------------------------------------------
    std::vector<double> alignedXZErrors;

    alignedXZErrors.reserve(
        commonFrames.size()
    );

    for (Eigen::Index i = 0;
        i < pairwiseCommonXZ.cols();
        ++i) {

        const Eigen::Vector3d pairwiseHomogeneous(
            pairwiseCommonXZ(0, i),
            pairwiseCommonXZ(1, i),
            1.0
        );

        const Eigen::Vector3d alignedPairwise =
            similarityTransform *
            pairwiseHomogeneous;

        const double dx =
            alignedPairwise.x() -
            mapBasedCommonXZ(0, i);

        const double dz =
            alignedPairwise.y() -
            mapBasedCommonXZ(1, i);

        alignedXZErrors.push_back(
            std::sqrt(
                dx * dx +
                dz * dz
            )
        );
    }


    const double meanAlignedError =
        computeMean(
            alignedXZErrors
        );

    const double medianAlignedError =
        computeMedian(
            alignedXZErrors
        );

    const double rmseAlignedError =
        computeRmse(
            alignedXZErrors
        );


    const auto maximumErrorIterator =
        std::max_element(
            alignedXZErrors.begin(),
            alignedXZErrors.end()
        );

    const double maximumAlignedError =
        *maximumErrorIterator;

    const std::size_t maximumErrorIndex =
        static_cast<std::size_t>(
            std::distance(
                alignedXZErrors.begin(),
                maximumErrorIterator
            )
            );


    const double alignmentScale =
        similarityTransform
        .block<2, 2>(0, 0)
        .col(0)
        .norm();

    const double alignmentRotationRadians =
        std::atan2(
            similarityTransform(1, 0),
            similarityTransform(0, 0)
        );

    const double alignmentRotationDegrees =
        alignmentRotationRadians *
        180.0 /
        CV_PI;


    std::cout
        << "\n=== Common-frame aligned comparison ===\n";

    std::cout
        << "Common frames: "
        << commonFrames.size()
        << '\n';

    std::cout
        << "Mean aligned X/Z divergence: "
        << meanAlignedError
        << '\n';

    std::cout
        << "Median aligned X/Z divergence: "
        << medianAlignedError
        << '\n';

    std::cout
        << "Aligned X/Z RMSE: "
        << rmseAlignedError
        << '\n';

    std::cout
        << "Maximum aligned X/Z divergence: "
        << maximumAlignedError
        << " at frame "
        << commonFrames[maximumErrorIndex]
        << '\n';

    std::cout
        << "Alignment scale: "
        << alignmentScale
        << '\n';

    std::cout
        << "Alignment rotation: "
        << alignmentRotationDegrees
        << " deg\n\n";

    const double pairwisePathLength =
        computePathLength(
            pairwiseTrajectory
        );

    const double mapBasedPathLength =
        computePathLength(
            mapBasedTrajectory
        );

    const double pairwiseEndDisplacement =
        computeEndDisplacement(
            pairwiseTrajectory
        );

    const double mapBasedEndDisplacement =
        computeEndDisplacement(
            mapBasedTrajectory
        );


    std::cout
        << "Pairwise trajectory points: "
        << pairwiseTrajectory.size()
        << '\n';

    std::cout
        << "Map-based trajectory points: "
        << mapBasedTrajectory.size()
        << '\n';

    std::cout
        << "Pairwise path length: "
        << pairwisePathLength
        << '\n';

    std::cout
        << "Pairwise start-to-end displacement: "
        << pairwiseEndDisplacement
        << '\n';

    std::cout
        << "Map-based path length: "
        << mapBasedPathLength
        << '\n';

    std::cout
        << "Map-based start-to-end displacement: "
        << mapBasedEndDisplacement
        << '\n';


    if (!normalizeTrajectory(
        pairwiseTrajectory)) {

        std::cerr
            << "Error: could not normalize pairwise trajectory.\n";

        return 1;
    }

    if (!normalizeTrajectory(
        mapBasedTrajectory)) {

        std::cerr
            << "Error: could not normalize map-based trajectory.\n";

        return 1;
    }


    std::cout
        << "Pairwise normalized path ratio: "
        << pairwisePathLength /
        pairwiseEndDisplacement
        << '\n';

    std::cout
        << "Map-based normalized path ratio: "
        << mapBasedPathLength /
        mapBasedEndDisplacement
        << '\n';


    double minX =
        std::numeric_limits<double>::infinity();

    double maxX =
        -std::numeric_limits<double>::infinity();

    double minZ =
        std::numeric_limits<double>::infinity();

    double maxZ =
        -std::numeric_limits<double>::infinity();


    auto updateBounds =
        [&](const std::vector<TrajectoryPoint>& trajectory) {

        for (const TrajectoryPoint& point : trajectory) {
            minX =
                std::min(minX, point.x);

            maxX =
                std::max(maxX, point.x);

            minZ =
                std::min(minZ, point.z);

            maxZ =
                std::max(maxZ, point.z);
        }
        };


    updateBounds(
        pairwiseTrajectory
    );

    updateBounds(
        mapBasedTrajectory
    );


    double rangeX =
        maxX - minX;

    double rangeZ =
        maxZ - minZ;

    if (rangeX < 1e-9) {
        rangeX = 1.0;
    }

    if (rangeZ < 1e-9) {
        rangeZ = 1.0;
    }


    const int imageWidth = 1000;
    const int imageHeight = 800;
    const int margin = 80;

    cv::Mat visualization(
        imageHeight,
        imageWidth,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );


    const double scaleX =
        static_cast<double>(
            imageWidth - 2 * margin
            ) / rangeX;

    const double scaleZ =
        static_cast<double>(
            imageHeight - 2 * margin
            ) / rangeZ;

    const double scale =
        std::min(
            scaleX,
            scaleZ
        );

    const double drawnWidth =
        rangeX * scale;

    const double drawnHeight =
        rangeZ * scale;

    const double offsetX =
        (imageWidth - drawnWidth) /
        2.0;

    const double offsetY =
        (imageHeight - drawnHeight) /
        2.0;


    auto worldToImage =
        [&](double x, double z) {

        const int pixelX =
            static_cast<int>(
                offsetX +
                (x - minX) * scale
                );

        const int pixelY =
            static_cast<int>(
                imageHeight -
                (
                    offsetY +
                    (z - minZ) * scale
                    )
                );

        return cv::Point(
            pixelX,
            pixelY
        );
        };


    auto drawTrajectory =
        [&](const std::vector<TrajectoryPoint>& trajectory,
            const cv::Scalar& color) {

                for (std::size_t i = 1;
                    i < trajectory.size();
                    ++i) {

                    const cv::Point previous =
                        worldToImage(
                            trajectory[i - 1].x,
                            trajectory[i - 1].z
                        );

                    const cv::Point current =
                        worldToImage(
                            trajectory[i].x,
                            trajectory[i].z
                        );

                    cv::line(
                        visualization,
                        previous,
                        current,
                        color,
                        3,
                        cv::LINE_AA
                    );
                }
        };


    drawTrajectory(
        pairwiseTrajectory,
        cv::Scalar(220, 80, 20)
    );

    drawTrajectory(
        mapBasedTrajectory,
        cv::Scalar(20, 80, 220)
    );


    const cv::Point startPoint =
        worldToImage(
            0.0,
            0.0
        );

    cv::circle(
        visualization,
        startPoint,
        8,
        cv::Scalar(0, 160, 0),
        cv::FILLED,
        cv::LINE_AA
    );


    cv::putText(
        visualization,
        "Normalized X/Z trajectory comparison",
        cv::Point(25, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );


    cv::line(
        visualization,
        cv::Point(30, 65),
        cv::Point(60, 65),
        cv::Scalar(220, 80, 20),
        3,
        cv::LINE_AA
    );

    cv::putText(
        visualization,
        "Pairwise VO",
        cv::Point(70, 72),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );


    cv::line(
        visualization,
        cv::Point(30, 95),
        cv::Point(60, 95),
        cv::Scalar(20, 80, 220),
        3,
        cv::LINE_AA
    );

    cv::putText(
        visualization,
        "Map-based VO",
        cv::Point(70, 102),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );


    if (outputImagePath.has_parent_path()) {
        fs::create_directories(
            outputImagePath.parent_path()
        );
    }

    if (!cv::imwrite(
        outputImagePath.string(),
        visualization)) {

        std::cerr
            << "Error: could not save comparison image.\n";

        return 1;
    }


    std::cout
        << "Comparison saved to: "
        << outputImagePath.string()
        << '\n';


    // ------------------------------------------------------------
// Courbe de divergence X/Z après alignement
// ------------------------------------------------------------
    const int errorPlotWidth = 1000;
    const int errorPlotHeight = 500;
    const int errorPlotMargin = 70;

    cv::Mat errorVisualization(
        errorPlotHeight,
        errorPlotWidth,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    const int minimumCommonFrame =
        commonFrames.front();

    const int maximumCommonFrame =
        commonFrames.back();

    double commonFrameRange =
        static_cast<double>(
            maximumCommonFrame -
            minimumCommonFrame
            );

    if (commonFrameRange < 1e-9) {
        commonFrameRange = 1.0;
    }

    double maximumPlotError =
        maximumAlignedError;

    if (maximumPlotError < 1e-9) {
        maximumPlotError = 1.0;
    }


    auto errorToImage =
        [&](int frameIndex, double error) {

        const int pixelX =
            static_cast<int>(
                errorPlotMargin +
                (
                    frameIndex -
                    minimumCommonFrame
                    ) *
                static_cast<double>(
                    errorPlotWidth -
                    2 * errorPlotMargin
                    ) /
                commonFrameRange
                );

        const int pixelY =
            static_cast<int>(
                errorPlotHeight -
                (
                    errorPlotMargin +
                    error /
                    maximumPlotError *
                    (
                        errorPlotHeight -
                        2 * errorPlotMargin
                        )
                    )
                );

        return cv::Point(
            pixelX,
            pixelY
        );
        };


    for (std::size_t i = 1;
        i < alignedXZErrors.size();
        ++i) {

        const cv::Point previous =
            errorToImage(
                commonFrames[i - 1],
                alignedXZErrors[i - 1]
            );

        const cv::Point current =
            errorToImage(
                commonFrames[i],
                alignedXZErrors[i]
            );

        cv::line(
            errorVisualization,
            previous,
            current,
            cv::Scalar(60, 60, 60),
            2,
            cv::LINE_AA
        );

        cv::circle(
            errorVisualization,
            current,
            3,
            cv::Scalar(0, 0, 0),
            cv::FILLED,
            cv::LINE_AA
        );
    }


    // Ligne horizontale = divergence moyenne
    const int meanErrorPixelY =
        errorToImage(
            minimumCommonFrame,
            meanAlignedError
        ).y;

    cv::line(
        errorVisualization,
        cv::Point(
            errorPlotMargin,
            meanErrorPixelY
        ),
        cv::Point(
            errorPlotWidth -
            errorPlotMargin,
            meanErrorPixelY
        ),
        cv::Scalar(180, 180, 180),
        1,
        cv::LINE_AA
    );


    cv::putText(
        errorVisualization,
        "Pairwise vs map-based - aligned X/Z divergence",
        cv::Point(25, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );

    cv::putText(
        errorVisualization,
        "frame",
        cv::Point(
            errorPlotWidth - 90,
            errorPlotHeight - 20
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );


    const fs::path alignedErrorImagePath =
        outputImagePath.parent_path() /
        (
            outputImagePath.stem().string() +
            "_aligned_error" +
            outputImagePath.extension().string()
            );

    if (!cv::imwrite(
        alignedErrorImagePath.string(),
        errorVisualization)) {

        std::cerr
            << "Error: could not save aligned error plot.\n";

        return 1;
    }

    std::cout
        << "Aligned divergence plot saved to: "
        << alignedErrorImagePath.string()
        << '\n';

    cv::imshow(
        "VO trajectory comparison",
        visualization
    );

    cv::imshow(
        "Aligned X/Z divergence",
        errorVisualization
    );

    cv::waitKey(0);

    return 0;
}