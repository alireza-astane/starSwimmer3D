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
    sphere { m*<-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 1 }        
    sphere {  m*<-8.092132236122156e-18,-2.6636331307281875e-18,8.684402272185087>, 1 }
    sphere {  m*<9.428090415820634,-1.845675551226533e-18,-3.0389310611482587>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0389310611482587>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0389310611482587>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-8.092132236122156e-18,-2.6636331307281875e-18,8.684402272185087>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5 }
    cylinder { m*<9.428090415820634,-1.845675551226533e-18,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5}

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
    sphere { m*<-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 1 }        
    sphere {  m*<-8.092132236122156e-18,-2.6636331307281875e-18,8.684402272185087>, 1 }
    sphere {  m*<9.428090415820634,-1.845675551226533e-18,-3.0389310611482587>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.0389310611482587>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.0389310611482587>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-8.092132236122156e-18,-2.6636331307281875e-18,8.684402272185087>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5 }
    cylinder { m*<9.428090415820634,-1.845675551226533e-18,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.0389310611482587>, <-4.683965729105083e-18,6.967718167810245e-19,0.29440227218507387>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    