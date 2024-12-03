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
    sphere { m*<0.6895566084415424,0.938544392062262,0.27358067692034904>, 1 }        
    sphere {  m*<0.9319410831280488,1.0251136734225597,3.262516134210273>, 1 }
    sphere {  m*<3.4251882721905833,1.0251136734225592,-0.9547660742803419>, 1 }
    sphere {  m*<-1.8091211577875015,4.636307060680105,-1.2037883467790214>, 1}
    sphere { m*<-3.922314460373022,-7.5047852499366705,-2.4525887388203502>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9319410831280488,1.0251136734225597,3.262516134210273>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5 }
    cylinder { m*<3.4251882721905833,1.0251136734225592,-0.9547660742803419>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5}
    cylinder { m*<-1.8091211577875015,4.636307060680105,-1.2037883467790214>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5 }
    cylinder {  m*<-3.922314460373022,-7.5047852499366705,-2.4525887388203502>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5}

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
    sphere { m*<0.6895566084415424,0.938544392062262,0.27358067692034904>, 1 }        
    sphere {  m*<0.9319410831280488,1.0251136734225597,3.262516134210273>, 1 }
    sphere {  m*<3.4251882721905833,1.0251136734225592,-0.9547660742803419>, 1 }
    sphere {  m*<-1.8091211577875015,4.636307060680105,-1.2037883467790214>, 1}
    sphere { m*<-3.922314460373022,-7.5047852499366705,-2.4525887388203502>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9319410831280488,1.0251136734225597,3.262516134210273>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5 }
    cylinder { m*<3.4251882721905833,1.0251136734225592,-0.9547660742803419>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5}
    cylinder { m*<-1.8091211577875015,4.636307060680105,-1.2037883467790214>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5 }
    cylinder {  m*<-3.922314460373022,-7.5047852499366705,-2.4525887388203502>, <0.6895566084415424,0.938544392062262,0.27358067692034904>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    