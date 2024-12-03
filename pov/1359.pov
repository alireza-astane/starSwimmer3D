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
    sphere { m*<0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 1 }        
    sphere {  m*<0.5682215606404699,-8.177454076454531e-19,4.008960349741342>, 1 }
    sphere {  m*<7.4859429912548645,3.5725113899685445e-18,-1.6528876691220085>, 1 }
    sphere {  m*<-4.2971005727549,8.164965809277259,-2.2089231305979853>, 1}
    sphere { m*<-4.2971005727549,-8.164965809277259,-2.208923130597988>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5682215606404699,-8.177454076454531e-19,4.008960349741342>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5 }
    cylinder { m*<7.4859429912548645,3.5725113899685445e-18,-1.6528876691220085>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5}
    cylinder { m*<-4.2971005727549,8.164965809277259,-2.2089231305979853>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5 }
    cylinder {  m*<-4.2971005727549,-8.164965809277259,-2.208923130597988>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5}

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
    sphere { m*<0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 1 }        
    sphere {  m*<0.5682215606404699,-8.177454076454531e-19,4.008960349741342>, 1 }
    sphere {  m*<7.4859429912548645,3.5725113899685445e-18,-1.6528876691220085>, 1 }
    sphere {  m*<-4.2971005727549,8.164965809277259,-2.2089231305979853>, 1}
    sphere { m*<-4.2971005727549,-8.164965809277259,-2.208923130597988>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5682215606404699,-8.177454076454531e-19,4.008960349741342>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5 }
    cylinder { m*<7.4859429912548645,3.5725113899685445e-18,-1.6528876691220085>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5}
    cylinder { m*<-4.2971005727549,8.164965809277259,-2.2089231305979853>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5 }
    cylinder {  m*<-4.2971005727549,-8.164965809277259,-2.208923130597988>, <0.49591724412180155,-4.8753733396497315e-18,1.009829361665442>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    