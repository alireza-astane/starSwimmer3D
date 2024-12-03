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
    sphere { m*<1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 1 }        
    sphere {  m*<1.6778867757150589e-18,-6.141377183033782e-18,6.432312325228807>, 1 }
    sphere {  m*<9.428090415820634,1.6006455840510642e-19,-2.561021008104548>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.561021008104548>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.561021008104548>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.6778867757150589e-18,-6.141377183033782e-18,6.432312325228807>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5 }
    cylinder { m*<9.428090415820634,1.6006455840510642e-19,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5}

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
    sphere { m*<1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 1 }        
    sphere {  m*<1.6778867757150589e-18,-6.141377183033782e-18,6.432312325228807>, 1 }
    sphere {  m*<9.428090415820634,1.6006455840510642e-19,-2.561021008104548>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.561021008104548>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.561021008104548>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.6778867757150589e-18,-6.141377183033782e-18,6.432312325228807>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5 }
    cylinder { m*<9.428090415820634,1.6006455840510642e-19,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.561021008104548>, <1.8469047138634507e-18,-5.4105415402874195e-18,0.7723123252287857>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    