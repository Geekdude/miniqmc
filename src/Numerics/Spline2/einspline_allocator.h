/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////
/** @file einspline_allocator.h
 *
 * Rename aligned_alloc/aligned_free as einspline_alloc/einspline_free to
 * avoid naming conflicts with the standards
 */

#ifndef EINSPLINE_ALIGNED_ALLOC_H
#define EINSPLINE_ALIGNED_ALLOC_H

//#ifdef __cplusplus
//extern "C" {
  //#endif

void *einspline_alloc(size_t size, size_t alignment);

void einspline_free(void *ptr);

multi_UBspline_3d_s *
einspline_create_multi_UBspline_3d_s(Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                     BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
                                     int num_splines);

multi_UBspline_3d_d *
einspline_create_multi_UBspline_3d_d(Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                     BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                     int num_splines);

  UBspline_3d_s *einspline_create_UBspline_3d_s(Ugrid x_grid, Ugrid y_grid,
						Ugrid z_grid, BCtype_s xBC,
						BCtype_s yBC, BCtype_s zBC,
						float *data);

  UBspline_3d_d *einspline_create_UBspline_3d_d(Ugrid x_grid, Ugrid y_grid,
						Ugrid z_grid, BCtype_d xBC,
						BCtype_d yBC, BCtype_d zBC,
						double *data);



//#ifdef __cplusplus
//}
//#endif

#endif
