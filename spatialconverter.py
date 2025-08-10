from rubicon.objc import ObjCClass, objc_method, objc_property, ObjCInstance, NSDictionary, objc_id, objc_const
from rubicon.objc.runtime import load_library
from ctypes import c_double
import math

ImageIO = load_library("ImageIO")

NSURL = ObjCClass("NSURL")
NSString = ObjCClass("NSString")
CFDictionaryRef = ObjCClass("NSDictionary")
NSMutableDictionary = ObjCClass("NSMutableDictionary")
CFNumber = ObjCClass("NSNumber")
CFArray = ObjCClass("NSArray")
CFBoolean = ObjCClass("NSNumber")

# ImageIO functions (from CoreFoundation C API)
from ctypes import cdll, c_void_p, c_uint, POINTER
from ctypes.util import find_library

corefoundation = cdll.LoadLibrary(find_library("CoreFoundation"))
imageio = cdll.LoadLibrary(find_library("ImageIO"))

# Function prototypes (C)
corefoundation.CFRelease.restype = None
corefoundation.CFRelease.argtypes = [c_void_p]

# CGImageSourceRef CGImageSourceCreateWithURL(CFURLRef url, CFDictionaryRef options);
imageio.CGImageSourceCreateWithURL.restype = c_void_p
imageio.CGImageSourceCreateWithURL.argtypes = [c_void_p, c_void_p]

# size_t CGImageSourceGetPrimaryImageIndex(CGImageSourceRef isrc);
imageio.CGImageSourceGetPrimaryImageIndex.restype = c_uint
imageio.CGImageSourceGetPrimaryImageIndex.argtypes = [c_void_p]

# CFDictionaryRef CGImageSourceCopyPropertiesAtIndex(CGImageSourceRef isrc, size_t index, CFDictionaryRef options);
imageio.CGImageSourceCopyPropertiesAtIndex.restype = c_void_p
imageio.CGImageSourceCopyPropertiesAtIndex.argtypes = [c_void_p, c_uint, c_void_p]

# CGImageDestinationRef CGImageDestinationCreateWithURL(CFURLRef url, CFStringRef type, size_t count, CFDictionaryRef options);
imageio.CGImageDestinationCreateWithURL.restype = c_void_p
imageio.CGImageDestinationCreateWithURL.argtypes = [c_void_p, c_void_p, c_uint, c_void_p]

# void CGImageDestinationAddImageFromSource(CGImageDestinationRef dst, CGImageSourceRef src, size_t index, CFDictionaryRef properties);
imageio.CGImageDestinationAddImageFromSource.restype = None
imageio.CGImageDestinationAddImageFromSource.argtypes = [c_void_p, c_void_p, c_uint, c_void_p]

# bool CGImageDestinationFinalize(CGImageDestinationRef dst);
imageio.CGImageDestinationFinalize.restype = c_void_p
imageio.CGImageDestinationFinalize.argtypes = [c_void_p]

# Some constants as strings (bridge to CFStringRef automatically)
kCGImagePropertyPixelWidth = objc_const(ImageIO, "kCGImagePropertyPixelWidth")
kCGImagePropertyPixelHeight = objc_const(ImageIO, "kCGImagePropertyPixelHeight")
kCGImagePropertyGroups = objc_const(ImageIO, "kCGImagePropertyGroups")
kCGImagePropertyGroupIndex = objc_const(ImageIO, "kCGImagePropertyGroupIndex")
kCGImagePropertyGroupType = objc_const(ImageIO, "kCGImagePropertyGroupType")
kCGImagePropertyGroupImageIsLeftImage = objc_const(ImageIO, "kCGImagePropertyGroupImageIsLeftImage")
kCGImagePropertyGroupImageIsRightImage = objc_const(ImageIO, "kCGImagePropertyGroupImageIsRightImage")
kCGImagePropertyGroupImageDisparityAdjustment = objc_const(ImageIO, "kCGImagePropertyGroupImageDisparityAdjustment")
kCGImagePropertyHEIFDictionary = objc_const(ImageIO, "kCGImagePropertyHEIFDictionary")
kCGImagePropertyHasAlpha = objc_const(ImageIO, "kCGImagePropertyHasAlpha")
kCGImagePropertyPrimaryImage = objc_const(ImageIO, "kCGImagePropertyPrimaryImage")
kCGImagePropertyGroupTypeStereoPair = objc_const(ImageIO, "kCGImagePropertyGroupTypeStereoPair")

kIIOMetadata_CameraExtrinsicsKey = objc_const(ImageIO, "kIIOMetadata_CameraExtrinsicsKey")
kIIOCameraExtrinsics_Position = objc_const(ImageIO, "kIIOCameraExtrinsics_Position")
kIIOCameraExtrinsics_Rotation = objc_const(ImageIO, "kIIOCameraExtrinsics_Rotation")
kIIOMetadata_CameraModelKey = objc_const(ImageIO, "kIIOMetadata_CameraModelKey")
kIIOCameraModel_Intrinsics = objc_const(ImageIO, "kIIOCameraModel_Intrinsics")
kIIOCameraModel_ModelType = objc_const(ImageIO, "kIIOCameraModel_ModelType")
kIIOCameraModelType_SimplifiedPinhole = objc_const(ImageIO, "kIIOCameraModelType_SimplifiedPinhole")

def objc_str(py_str: str) -> NSString:
    return NSString.stringWithString_(py_str)

kUTTypeHEIC = objc_str("public.heic")



IDENTITY_ROTATION = [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]


class ConversionError(Exception):
    pass


class SpatialPhotoConverter:
    def __init__(self, left_image_path, right_image_path, output_image_path, baseline_mm, focal_length, disparity_adjustment):
        self.leftImageURL = NSURL.fileURLWithPath_(left_image_path)
        self.rightImageURL = NSURL.fileURLWithPath_(right_image_path)
        self.outputImageURL = NSURL.fileURLWithPath_(output_image_path)
        self.baselineInMillimeters = float(baseline_mm)
        self.focal_length = float(focal_length)
        self.disparityAdjustment = float(disparity_adjustment)

    class StereoPairImage:
        def __init__(self, url: NSURL):
            # Create CGImageSource from URL
            src = imageio.CGImageSourceCreateWithURL(url.ptr, None)
            if not src:
                raise ConversionError("Could not open image URL as image source")
            self.source = src
            self.primaryImageIndex = imageio.CGImageSourceGetPrimaryImageIndex(src)

            props_ptr = imageio.CGImageSourceCopyPropertiesAtIndex(src, self.primaryImageIndex, None)
            if not props_ptr:
                raise ConversionError("Could not copy image properties")

            # Try to convert CFDictionaryRef to NSDictionary
            props = ObjCInstance(props_ptr)

            width = props.get(kCGImagePropertyPixelWidth)
            height = props.get(kCGImagePropertyPixelHeight)
            if width is None or height is None:
                raise ConversionError("Unable to read image size")
            self.width = int(str(width))
            self.height = int(str(height))

        def intrinsics(self, focalLength):
            w = float(self.width)
            h = float(self.height)
            focalLengthX = focalLength
            focalLengthY = focalLengthX
            principalPointX = 0.5 * w
            principalPointY = 0.5 * h
            return [
                focalLengthX, 0.0, principalPointX,
                0.0, focalLengthY, principalPointY,
                0.0, 0.0, 1.0
            ]

    def properties_dictionary(self, is_left, encoded_disparity_adjustment, position, intrinsics):
        # Build the stereo group dict
        group_dict = NSMutableDictionary.alloc().init()
        group_dict[kCGImagePropertyGroupIndex] = 0
        group_dict[kCGImagePropertyGroupType] = kCGImagePropertyGroupTypeStereoPair
        if is_left:
            group_dict[kCGImagePropertyGroupImageIsLeftImage] = True
        else:
            group_dict[kCGImagePropertyGroupImageIsRightImage] = True
        group_dict[kCGImagePropertyGroupImageDisparityAdjustment] = encoded_disparity_adjustment

        # HEIF dictionary
        camera_extrinsics = NSMutableDictionary.alloc().init()
        camera_extrinsics[kIIOCameraExtrinsics_Position] = position
        camera_extrinsics[kIIOCameraExtrinsics_Rotation] = IDENTITY_ROTATION

        camera_model = NSMutableDictionary.alloc().init()
        camera_model[kIIOCameraModel_Intrinsics] = intrinsics
        camera_model[kIIOCameraModel_ModelType] = kIIOCameraModelType_SimplifiedPinhole

        heif_dict = NSMutableDictionary.alloc().init()
        heif_dict[kIIOMetadata_CameraExtrinsicsKey] = camera_extrinsics
        heif_dict[kIIOMetadata_CameraModelKey] = camera_model

        props = NSMutableDictionary.alloc().init()
        props[kCGImagePropertyGroups] = group_dict
        props[kCGImagePropertyHEIFDictionary] = heif_dict
        props[kCGImagePropertyHasAlpha] = False

        return props

    def convert(self):
        left_image = self.StereoPairImage(self.leftImageURL)
        right_image = self.StereoPairImage(self.rightImageURL)

        if left_image.width != right_image.width or left_image.height != right_image.height:
            raise ConversionError("Left and right image sizes do not match")

        baseline_m = self.baselineInMillimeters / 1000.0
        left_position = [0.0, 0.0, 0.0]
        right_position = [baseline_m, 0.0, 0.0]

        intrinsics = left_image.intrinsics(self.focal_length)

        encoded_disparity_adjustment = int(self.disparityAdjustment * 1e4)

        left_props = self.properties_dictionary(True, encoded_disparity_adjustment, left_position, intrinsics)
        right_props = self.properties_dictionary(False, encoded_disparity_adjustment, right_position, intrinsics)

        destinationProperties = NSDictionary.dictionaryWithObjects([kCGImagePropertyPrimaryImage], forKeys=[0])
        print(type(self.outputImageURL))
        dest = imageio.CGImageDestinationCreateWithURL(
            self.outputImageURL.ptr,
            kUTTypeHEIC.ptr,
            2,
            destinationProperties.ptr
        )
        print("====")
        if not dest:
            raise ConversionError("Unable to create image destination")
        imageio.CGImageDestinationAddImageFromSource(
            dest,
            left_image.source,
            left_image.primaryImageIndex,
            left_props.ptr
        )
        imageio.CGImageDestinationAddImageFromSource(
            dest,
            right_image.source,
            right_image.primaryImageIndex,
            right_props.ptr
        )

        if not imageio.CGImageDestinationFinalize(dest):
            raise ConversionError("Unable to finalize image destination")


# Usage example (replace paths accordingly):
# converter = SpatialPhotoConverter(
#     "/path/to/left.jpg",
#     "/path/to/right.jpg",
#     "/path/to/output.heic",
#     baseline_mm=65.0,
#     horizontal_fov=90.0,
#     disparity_adjustment=0.0
# )
# converter.convert()

