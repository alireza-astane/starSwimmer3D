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
    sphere { m*<-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 1 }        
    sphere {  m*<0.42428023076596166,0.23162548463407284,6.9096733914274155>, 1 }
    sphere {  m*<2.501460808687746,-0.01789066684941424,-2.4795440655443564>, 1 }
    sphere {  m*<-1.8548629452114012,2.208549302182811,-2.2242803055091427>, 1}
    sphere { m*<-1.5870757241735693,-2.6791426402210865,-2.03473402034657>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42428023076596166,0.23162548463407284,6.9096733914274155>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5 }
    cylinder { m*<2.501460808687746,-0.01789066684941424,-2.4795440655443564>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5}
    cylinder { m*<-1.8548629452114012,2.208549302182811,-2.2242803055091427>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5 }
    cylinder {  m*<-1.5870757241735693,-2.6791426402210865,-2.03473402034657>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5}

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
    sphere { m*<-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 1 }        
    sphere {  m*<0.42428023076596166,0.23162548463407284,6.9096733914274155>, 1 }
    sphere {  m*<2.501460808687746,-0.01789066684941424,-2.4795440655443564>, 1 }
    sphere {  m*<-1.8548629452114012,2.208549302182811,-2.2242803055091427>, 1}
    sphere { m*<-1.5870757241735693,-2.6791426402210865,-2.03473402034657>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42428023076596166,0.23162548463407284,6.9096733914274155>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5 }
    cylinder { m*<2.501460808687746,-0.01789066684941424,-2.4795440655443564>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5}
    cylinder { m*<-1.8548629452114012,2.208549302182811,-2.2242803055091427>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5 }
    cylinder {  m*<-1.5870757241735693,-2.6791426402210865,-2.03473402034657>, <-0.23324758531851125,-0.11992464223578844,-1.2503345400931754>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    