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
    sphere { m*<-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 1 }        
    sphere {  m*<0.25837988715087934,0.14292613926958245,4.850827991087505>, 1 }
    sphere {  m*<2.547888827454835,0.006932276741141927,-1.903366157755947>, 1 }
    sphere {  m*<-1.808434926444312,2.233372245773367,-1.6481023977207339>, 1}
    sphere { m*<-1.5406477054064802,-2.6543196966305302,-1.458556112558161>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25837988715087934,0.14292613926958245,4.850827991087505>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5 }
    cylinder { m*<2.547888827454835,0.006932276741141927,-1.903366157755947>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5}
    cylinder { m*<-1.808434926444312,2.233372245773367,-1.6481023977207339>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5 }
    cylinder {  m*<-1.5406477054064802,-2.6543196966305302,-1.458556112558161>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5}

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
    sphere { m*<-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 1 }        
    sphere {  m*<0.25837988715087934,0.14292613926958245,4.850827991087505>, 1 }
    sphere {  m*<2.547888827454835,0.006932276741141927,-1.903366157755947>, 1 }
    sphere {  m*<-1.808434926444312,2.233372245773367,-1.6481023977207339>, 1}
    sphere { m*<-1.5406477054064802,-2.6543196966305302,-1.458556112558161>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25837988715087934,0.14292613926958245,4.850827991087505>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5 }
    cylinder { m*<2.547888827454835,0.006932276741141927,-1.903366157755947>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5}
    cylinder { m*<-1.808434926444312,2.233372245773367,-1.6481023977207339>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5 }
    cylinder {  m*<-1.5406477054064802,-2.6543196966305302,-1.458556112558161>, <-0.18681956655142207,-0.09510169864523224,-0.6741566323047647>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    