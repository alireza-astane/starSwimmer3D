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
    sphere { m*<-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 1 }        
    sphere {  m*<0.7977695358108853,0.11611526100055336,9.311964433422357>, 1 }
    sphere {  m*<8.165556734133682,-0.1689769897917086,-5.25871299565157>, 1 }
    sphere {  m*<-6.730406459555303,6.354104383828934,-3.767906092469965>, 1}
    sphere { m*<-3.0005918755922902,-6.055243448340954,-1.6391007442088203>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7977695358108853,0.11611526100055336,9.311964433422357>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5 }
    cylinder { m*<8.165556734133682,-0.1689769897917086,-5.25871299565157>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5}
    cylinder { m*<-6.730406459555303,6.354104383828934,-3.767906092469965>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5 }
    cylinder {  m*<-3.0005918755922902,-6.055243448340954,-1.6391007442088203>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5}

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
    sphere { m*<-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 1 }        
    sphere {  m*<0.7977695358108853,0.11611526100055336,9.311964433422357>, 1 }
    sphere {  m*<8.165556734133682,-0.1689769897917086,-5.25871299565157>, 1 }
    sphere {  m*<-6.730406459555303,6.354104383828934,-3.767906092469965>, 1}
    sphere { m*<-3.0005918755922902,-6.055243448340954,-1.6391007442088203>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7977695358108853,0.11611526100055336,9.311964433422357>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5 }
    cylinder { m*<8.165556734133682,-0.1689769897917086,-5.25871299565157>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5}
    cylinder { m*<-6.730406459555303,6.354104383828934,-3.767906092469965>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5 }
    cylinder {  m*<-3.0005918755922902,-6.055243448340954,-1.6391007442088203>, <-0.6213979583892765,-0.8738236528793639,-0.537325663612791>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    