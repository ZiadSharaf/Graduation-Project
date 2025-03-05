#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libpython-shared" for configuration "Release"
set_property(TARGET libpython-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libpython-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/libs/python313.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/python313.dll"
  )

list(APPEND _cmake_import_check_targets libpython-shared )
list(APPEND _cmake_import_check_files_for_libpython-shared "${_IMPORT_PREFIX}/libs/python313.lib" "${_IMPORT_PREFIX}/bin/python313.dll" )

# Import target "libpython3-shared" for configuration "Release"
set_property(TARGET libpython3-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libpython3-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/libs/python3.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/python3.dll"
  )

list(APPEND _cmake_import_check_targets libpython3-shared )
list(APPEND _cmake_import_check_files_for_libpython3-shared "${_IMPORT_PREFIX}/libs/python3.lib" "${_IMPORT_PREFIX}/bin/python3.dll" )

# Import target "python" for configuration "Release"
set_property(TARGET python APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(python PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/python.exe"
  )

list(APPEND _cmake_import_check_targets python )
list(APPEND _cmake_import_check_files_for_python "${_IMPORT_PREFIX}/bin/python.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
