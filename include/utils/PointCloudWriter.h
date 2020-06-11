/**
 * PointCloudWriter.h
 * Writes point cloud files to various file types
 *
 * Tim Garrett (garrettt@iastate.edu)
 * 2020.02.14
 */
#pragma once

// Eigen
#include <Eigen/Core>

// STL
#include <cmath>
#include <string>
#include <fstream>

// Local
#include "Types.h"

namespace texpert {
	class PointCloudWriter {
	public:
		/**
		 * Writes a point cloud to an OBJ file
		 * @param fileName The point cloud file name
		 * @param pointCloud The point cloud
		 * @param color The point cloud color scaled in range from 0.0 - 1.0
		 * @param writeInvalidPoints If invalid points should be written to the file
		 * @return If the file was successfully written
		 */
		static bool writeFileOBJ(const std::string fileName, const PointCloud& pointCloud, const Eigen::Vector3f& color=Eigen::Vector3f(1.0f, 0.0f, 0.0f), bool writeInvalidPoints=false)
		{
			// PointCloud.N isn't always updated, so use the raw points here
			if (pointCloud.points.size() == 0) return false;
			bool hasNormals = (pointCloud.points.size() == pointCloud.normals.size());

			// Open the file
			std::ofstream f;
			f.open(fileName, std::ios_base::out);
			if (!f.is_open()) return false;

			// Write each vertex
			for (int i = 0; i < pointCloud.points.size(); i++) {
				const Eigen::Vector3f& p = pointCloud.points.at(i);
				bool pointIsValid = std::isnormal(p.x()) && std::isnormal(p.y()) && std::isnormal(p.z());

				const Eigen::Vector3f* pN = (hasNormals ? &pointCloud.normals.at(i) : nullptr);
				bool normalIsValid = hasNormals && std::isnormal(pN->x()) && std::isnormal(pN->y()) && std::isnormal(pN->z());

				// Don't write invalid points if requested
				if (!writeInvalidPoints && !(pointIsValid)) continue;

				// Write the normal
				if (normalIsValid)
					f << "vn " << pN->x() << " " << pN->y() << " " << pN->z() << std::endl;
				else
					f << "vn 0.0 0.0 0.0" << std::endl;

				// Write the vertex
				f << "v " << p.x() << " " << p.y() << " " << p.z() << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
			}

			f.close();
			return true;
		}

		/**
		* Writes a point cloud to a ply file
		* @param fileName The ply file to write
		* @param pointCloud The point cloud to write
		* @param writeInvalidPoints If invalid points should be written
		* @param color The color of the point cloud
		* @param writeOrigin If the origin should be designated by a red point
		* @return If the write was successful
		*/
		static bool writePointCloudToPly(std::string fileName, PointCloud &pointCloud, bool writeInvalidPoints = false,
			const Eigen::Vector3f& color=Eigen::Vector3f(1.0f, 0.0f, 0.0f), bool writeOrigin = false)
		{
			auto isValid = [](Eigen::Vector3f& v)->bool {
				return (std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z()));
			};

			// Make sure there is a valid point cloud
			if (pointCloud.points.size() == 0) return false;

			// Calculate the number of points for the header
			int numPoints = 0;
			if (!writeInvalidPoints) {
				for (int i = 0; i < pointCloud.points.size(); i++) {
					if (isValid(pointCloud.points[i]))
						numPoints++;
				}
			}
			else {
				numPoints = pointCloud.points.size();
			}
			if (writeOrigin) numPoints += 3;

			// Open the file
			std::ofstream f;
			f.open(fileName, std::ios_base::out);
			if (!f.is_open()) return false;

			// Write the header
			f << "ply" << std::endl;
			f << "format ascii 1.0" << std::endl;
			f << "element vertex " << numPoints << std::endl;
			f << "property float x" << std::endl;
			f << "property float y" << std::endl;
			f << "property float z" << std::endl;
			f << "property float nx" << std::endl;
			f << "property float ny" << std::endl;
			f << "property float nz" << std::endl;
			f << "property uchar red" << std::endl;
			f << "property uchar green" << std::endl;
			f << "property uchar blue" << std::endl;
			f << "property uchar alpha" << std::endl;
			f << "element face 0" << std::endl;
			f << "property list uchar int vertex_indices" << std::endl;
			f << "end_header" << std::endl;

			// Optionally designate the origin with a red point
			if (writeOrigin) {
				f << "0.0 0.0 0.0 1.0 0.0 0.0 255 0 0 255" << std::endl;
				f << "0.0 0.0 0.0 0.0 1.0 0.0 255 0 0 255" << std::endl;
				f << "0.0 0.0 0.0 0.0 0.0 1.0 255 0 0 255" << std::endl;
			}

			// Write each vertex and color
			for (int i = 0; i < pointCloud.points.size(); i++) {
				Eigen::Vector3f& p = pointCloud.points[i];
				bool hasValidNormal = (i < pointCloud.normals.size()) && isValid(pointCloud.normals[i]);

				// Don't write invalid points if requested
				if (!writeInvalidPoints && !isValid(p)) continue;

				// Scale the color if necessary
				unsigned char r = static_cast<unsigned char>(color.x() * 255.0f);
				unsigned char g = static_cast<unsigned char>(color.y() * 255.0f);
				unsigned char b = static_cast<unsigned char>(color.z() * 255.0f);

				// Write the point
				f << p.x() << " " << p.y() << " " << p.z() << " ";

				// Write the normal
				if (hasValidNormal) {
					Eigen::Vector3f& n = pointCloud.normals[i];
					f << n.x() << " " << n.y() << " " << n.z() << " ";
				}
				else
					f << "0 0 0 ";

				// Write the color
				f << (int)r << " " << (int)g << " " << (int)b << " 255" << std::endl;
			}

			f.close();
			return true;
		}
	};
}