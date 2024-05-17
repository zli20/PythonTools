#include "MotionCapture.h"
#include <QtWidgets/QApplication>

#include "vtkGenericOpenGLRenderWindow.h"
#include "QVTKOpenGLNativeWidget.h"
#include "vtkAutoInit.h" 
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2)
VTK_MODULE_INIT(vtkRenderingContextOpenGL2)

int main(int argc, char *argv[])
{

    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());

    QApplication a(argc, argv);
    MotionCapture w;
    w.show();
    return a.exec();
}
