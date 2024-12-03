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
    sphere { m*<-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 1 }        
    sphere {  m*<0.37392822318156904,0.20470456323510444,6.284798258838586>, 1 }
    sphere {  m*<2.5162677694534374,-0.00997406042090812,-2.2957877067498944>, 1 }
    sphere {  m*<-1.8400559844457096,2.2164659086113163,-2.040523946714681>, 1}
    sphere { m*<-1.5722687634078778,-2.671226033792581,-1.8509776615521085>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37392822318156904,0.20470456323510444,6.284798258838586>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5 }
    cylinder { m*<2.5162677694534374,-0.00997406042090812,-2.2957877067498944>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5}
    cylinder { m*<-1.8400559844457096,2.2164659086113163,-2.040523946714681>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5 }
    cylinder {  m*<-1.5722687634078778,-2.671226033792581,-1.8509776615521085>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5}

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
    sphere { m*<-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 1 }        
    sphere {  m*<0.37392822318156904,0.20470456323510444,6.284798258838586>, 1 }
    sphere {  m*<2.5162677694534374,-0.00997406042090812,-2.2957877067498944>, 1 }
    sphere {  m*<-1.8400559844457096,2.2164659086113163,-2.040523946714681>, 1}
    sphere { m*<-1.5722687634078778,-2.671226033792581,-1.8509776615521085>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.37392822318156904,0.20470456323510444,6.284798258838586>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5 }
    cylinder { m*<2.5162677694534374,-0.00997406042090812,-2.2957877067498944>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5}
    cylinder { m*<-1.8400559844457096,2.2164659086113163,-2.040523946714681>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5 }
    cylinder {  m*<-1.5722687634078778,-2.671226033792581,-1.8509776615521085>, <-0.2184406245528197,-0.11200803580728234,-1.0665781812987138>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    