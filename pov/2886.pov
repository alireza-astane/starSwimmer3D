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
    sphere { m*<0.591869233337167,1.071524544677076,0.21582249887994664>, 1 }        
    sphere {  m*<0.833500147408988,1.1741043074601827,3.2043117745228056>, 1 }
    sphere {  m*<3.3267473364715214,1.1741043074601822,-1.0129704339678105>, 1 }
    sphere {  m*<-1.4468434560323762,4.031663029220684,-0.9895850126290071>, 1}
    sphere { m*<-3.9525252730474807,-7.421048779829424,-2.4704528474042986>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.833500147408988,1.1741043074601827,3.2043117745228056>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5 }
    cylinder { m*<3.3267473364715214,1.1741043074601822,-1.0129704339678105>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5}
    cylinder { m*<-1.4468434560323762,4.031663029220684,-0.9895850126290071>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5 }
    cylinder {  m*<-3.9525252730474807,-7.421048779829424,-2.4704528474042986>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5}

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
    sphere { m*<0.591869233337167,1.071524544677076,0.21582249887994664>, 1 }        
    sphere {  m*<0.833500147408988,1.1741043074601827,3.2043117745228056>, 1 }
    sphere {  m*<3.3267473364715214,1.1741043074601822,-1.0129704339678105>, 1 }
    sphere {  m*<-1.4468434560323762,4.031663029220684,-0.9895850126290071>, 1}
    sphere { m*<-3.9525252730474807,-7.421048779829424,-2.4704528474042986>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.833500147408988,1.1741043074601827,3.2043117745228056>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5 }
    cylinder { m*<3.3267473364715214,1.1741043074601822,-1.0129704339678105>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5}
    cylinder { m*<-1.4468434560323762,4.031663029220684,-0.9895850126290071>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5 }
    cylinder {  m*<-3.9525252730474807,-7.421048779829424,-2.4704528474042986>, <0.591869233337167,1.071524544677076,0.21582249887994664>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    