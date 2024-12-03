#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 1 }        
    sphere {  m*<-9.209138983239381e-18,-2.7519515878436518e-18,8.604061929510886>, 1 }
    sphere {  m*<9.428090415820634,-2.0095512089767496e-18,-3.0212714038224577>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0212714038224577>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0212714038224577>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.209138983239381e-18,-2.7519515878436518e-18,8.604061929510886>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0095512089767496e-18,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 1 }        
    sphere {  m*<-9.209138983239381e-18,-2.7519515878436518e-18,8.604061929510886>, 1 }
    sphere {  m*<9.428090415820634,-2.0095512089767496e-18,-3.0212714038224577>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0212714038224577>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0212714038224577>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-9.209138983239381e-18,-2.7519515878436518e-18,8.604061929510886>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0095512089767496e-18,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0212714038224577>, <-4.897062329953443e-18,1.220287282575747e-19,0.3120619295108745>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    