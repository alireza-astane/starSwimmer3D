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
    sphere { m*<-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 1 }        
    sphere {  m*<0.3627784831591571,0.2870847728356412,8.430638892753139>, 1 }
    sphere {  m*<4.113867893943406,0.01962538862646701,-3.7897942146797465>, 1 }
    sphere {  m*<-2.3107017852946656,2.1750617760246618,-2.4455005641411116>, 1}
    sphere { m*<-2.042914564256834,-2.7126301663792356,-2.255954278978541>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3627784831591571,0.2870847728356412,8.430638892753139>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5 }
    cylinder { m*<4.113867893943406,0.01962538862646701,-3.7897942146797465>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5}
    cylinder { m*<-2.3107017852946656,2.1750617760246618,-2.4455005641411116>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5 }
    cylinder {  m*<-2.042914564256834,-2.7126301663792356,-2.255954278978541>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5}

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
    sphere { m*<-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 1 }        
    sphere {  m*<0.3627784831591571,0.2870847728356412,8.430638892753139>, 1 }
    sphere {  m*<4.113867893943406,0.01962538862646701,-3.7897942146797465>, 1 }
    sphere {  m*<-2.3107017852946656,2.1750617760246618,-2.4455005641411116>, 1}
    sphere { m*<-2.042914564256834,-2.7126301663792356,-2.255954278978541>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3627784831591571,0.2870847728356412,8.430638892753139>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5 }
    cylinder { m*<4.113867893943406,0.01962538862646701,-3.7897942146797465>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5}
    cylinder { m*<-2.3107017852946656,2.1750617760246618,-2.4455005641411116>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5 }
    cylinder {  m*<-2.042914564256834,-2.7126301663792356,-2.255954278978541>, <-0.6691329155056902,-0.15366045272384546,-1.5061931128256867>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    