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
    sphere { m*<0.8483287011231705,0.7082897943393414,0.3674561161711586>, 1 }        
    sphere {  m*<1.0916440196951886,0.7695368720694754,3.356942959118423>, 1 }
    sphere {  m*<3.5848912087577256,0.7695368720694752,-0.8603392493721944>, 1 }
    sphere {  m*<-2.348205013262544,5.584226375785238,-1.5225336030741572>, 1}
    sphere { m*<-3.869393485489008,-7.655726618516805,-2.4212955888161716>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0916440196951886,0.7695368720694754,3.356942959118423>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5 }
    cylinder { m*<3.5848912087577256,0.7695368720694752,-0.8603392493721944>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5}
    cylinder { m*<-2.348205013262544,5.584226375785238,-1.5225336030741572>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5 }
    cylinder {  m*<-3.869393485489008,-7.655726618516805,-2.4212955888161716>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5}

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
    sphere { m*<0.8483287011231705,0.7082897943393414,0.3674561161711586>, 1 }        
    sphere {  m*<1.0916440196951886,0.7695368720694754,3.356942959118423>, 1 }
    sphere {  m*<3.5848912087577256,0.7695368720694752,-0.8603392493721944>, 1 }
    sphere {  m*<-2.348205013262544,5.584226375785238,-1.5225336030741572>, 1}
    sphere { m*<-3.869393485489008,-7.655726618516805,-2.4212955888161716>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0916440196951886,0.7695368720694754,3.356942959118423>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5 }
    cylinder { m*<3.5848912087577256,0.7695368720694752,-0.8603392493721944>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5}
    cylinder { m*<-2.348205013262544,5.584226375785238,-1.5225336030741572>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5 }
    cylinder {  m*<-3.869393485489008,-7.655726618516805,-2.4212955888161716>, <0.8483287011231705,0.7082897943393414,0.3674561161711586>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    