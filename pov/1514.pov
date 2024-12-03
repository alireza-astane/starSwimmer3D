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
    sphere { m*<0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 1 }        
    sphere {  m*<0.8098941755481646,-1.257588962163504e-18,3.9205392873711293>, 1 }
    sphere {  m*<6.648355453895841,3.532772581658653e-18,-1.4273377057161405>, 1 }
    sphere {  m*<-4.131510579596047,8.164965809277259,-2.237058809712069>, 1}
    sphere { m*<-4.131510579596047,-8.164965809277259,-2.237058809712072>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8098941755481646,-1.257588962163504e-18,3.9205392873711293>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5 }
    cylinder { m*<6.648355453895841,3.532772581658653e-18,-1.4273377057161405>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5}
    cylinder { m*<-4.131510579596047,8.164965809277259,-2.237058809712069>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5 }
    cylinder {  m*<-4.131510579596047,-8.164965809277259,-2.237058809712072>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5}

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
    sphere { m*<0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 1 }        
    sphere {  m*<0.8098941755481646,-1.257588962163504e-18,3.9205392873711293>, 1 }
    sphere {  m*<6.648355453895841,3.532772581658653e-18,-1.4273377057161405>, 1 }
    sphere {  m*<-4.131510579596047,8.164965809277259,-2.237058809712069>, 1}
    sphere { m*<-4.131510579596047,-8.164965809277259,-2.237058809712072>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8098941755481646,-1.257588962163504e-18,3.9205392873711293>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5 }
    cylinder { m*<6.648355453895841,3.532772581658653e-18,-1.4273377057161405>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5}
    cylinder { m*<-4.131510579596047,8.164965809277259,-2.237058809712069>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5 }
    cylinder {  m*<-4.131510579596047,-8.164965809277259,-2.237058809712072>, <0.7007214010717712,-5.241056078923601e-18,0.9225224949072929>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    