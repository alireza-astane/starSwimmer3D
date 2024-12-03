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
    sphere { m*<1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 1 }        
    sphere {  m*<1.352072636893273,1.5185730574829112e-18,3.6941727198529186>, 1 }
    sphere {  m*<4.692651306836238,7.029763917373472e-18,-0.8379410456664595>, 1 }
    sphere {  m*<-3.789681837962577,8.164965809277259,-2.297158050635434>, 1}
    sphere { m*<-3.789681837962577,-8.164965809277259,-2.2971580506354368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.352072636893273,1.5185730574829112e-18,3.6941727198529186>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5 }
    cylinder { m*<4.692651306836238,7.029763917373472e-18,-0.8379410456664595>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5}
    cylinder { m*<-3.789681837962577,8.164965809277259,-2.297158050635434>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5 }
    cylinder {  m*<-3.789681837962577,-8.164965809277259,-2.2971580506354368>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5}

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
    sphere { m*<1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 1 }        
    sphere {  m*<1.352072636893273,1.5185730574829112e-18,3.6941727198529186>, 1 }
    sphere {  m*<4.692651306836238,7.029763917373472e-18,-0.8379410456664595>, 1 }
    sphere {  m*<-3.789681837962577,8.164965809277259,-2.297158050635434>, 1}
    sphere { m*<-3.789681837962577,-8.164965809277259,-2.2971580506354368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.352072636893273,1.5185730574829112e-18,3.6941727198529186>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5 }
    cylinder { m*<4.692651306836238,7.029763917373472e-18,-0.8379410456664595>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5}
    cylinder { m*<-3.789681837962577,8.164965809277259,-2.297158050635434>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5 }
    cylinder {  m*<-3.789681837962577,-8.164965809277259,-2.2971580506354368>, <1.1441081016090633,3.9768983922698984e-20,0.701381162099728>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    