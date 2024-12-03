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
    sphere { m*<-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 1 }        
    sphere {  m*<0.4826508837707004,-0.26460658107013113,9.166617041450014>, 1 }
    sphere {  m*<7.838002321770678,-0.35352685706448694,-5.412876248595319>, 1 }
    sphere {  m*<-6.375017332418197,5.387851811032439,-3.4732529911240113>, 1}
    sphere { m*<-2.1828417046698276,-3.7708925020938384,-1.3022407229156368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4826508837707004,-0.26460658107013113,9.166617041450014>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5 }
    cylinder { m*<7.838002321770678,-0.35352685706448694,-5.412876248595319>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5}
    cylinder { m*<-6.375017332418197,5.387851811032439,-3.4732529911240113>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5 }
    cylinder {  m*<-2.1828417046698276,-3.7708925020938384,-1.3022407229156368>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5}

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
    sphere { m*<-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 1 }        
    sphere {  m*<0.4826508837707004,-0.26460658107013113,9.166617041450014>, 1 }
    sphere {  m*<7.838002321770678,-0.35352685706448694,-5.412876248595319>, 1 }
    sphere {  m*<-6.375017332418197,5.387851811032439,-3.4732529911240113>, 1}
    sphere { m*<-2.1828417046698276,-3.7708925020938384,-1.3022407229156368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4826508837707004,-0.26460658107013113,9.166617041450014>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5 }
    cylinder { m*<7.838002321770678,-0.35352685706448694,-5.412876248595319>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5}
    cylinder { m*<-6.375017332418197,5.387851811032439,-3.4732529911240113>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5 }
    cylinder {  m*<-2.1828417046698276,-3.7708925020938384,-1.3022407229156368>, <-0.9468256004984958,-1.1059754319142185,-0.6949289303648802>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    