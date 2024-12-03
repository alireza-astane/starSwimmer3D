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
    sphere { m*<0.2878181207325036,0.7519572952354898,0.03870954571037333>, 1 }        
    sphere {  m*<0.5285532254741953,0.8806673734158152,3.026264316830923>, 1 }
    sphere {  m*<3.022526514738759,0.8539912706218642,-1.1904999797408098>, 1 }
    sphere {  m*<-1.3337972391603872,3.080431239654092,-0.9352362197055958>, 1}
    sphere { m*<-3.2709447802711598,-5.975373748801855,-2.0232152213823817>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5285532254741953,0.8806673734158152,3.026264316830923>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5 }
    cylinder { m*<3.022526514738759,0.8539912706218642,-1.1904999797408098>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5}
    cylinder { m*<-1.3337972391603872,3.080431239654092,-0.9352362197055958>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5 }
    cylinder {  m*<-3.2709447802711598,-5.975373748801855,-2.0232152213823817>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5}

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
    sphere { m*<0.2878181207325036,0.7519572952354898,0.03870954571037333>, 1 }        
    sphere {  m*<0.5285532254741953,0.8806673734158152,3.026264316830923>, 1 }
    sphere {  m*<3.022526514738759,0.8539912706218642,-1.1904999797408098>, 1 }
    sphere {  m*<-1.3337972391603872,3.080431239654092,-0.9352362197055958>, 1}
    sphere { m*<-3.2709447802711598,-5.975373748801855,-2.0232152213823817>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5285532254741953,0.8806673734158152,3.026264316830923>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5 }
    cylinder { m*<3.022526514738759,0.8539912706218642,-1.1904999797408098>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5}
    cylinder { m*<-1.3337972391603872,3.080431239654092,-0.9352362197055958>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5 }
    cylinder {  m*<-3.2709447802711598,-5.975373748801855,-2.0232152213823817>, <0.2878181207325036,0.7519572952354898,0.03870954571037333>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    