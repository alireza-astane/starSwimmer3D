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
    sphere { m*<0.926130746962688,0.5891322197174367,0.41345754924973677>, 1 }        
    sphere {  m*<1.1697744580887686,0.6384286518924156,3.4031390434737325>, 1 }
    sphere {  m*<3.663021647151304,0.6384286518924154,-0.8141431650168849>, 1 }
    sphere {  m*<-2.5994622370329497,6.044817556187686,-1.671095791362063>, 1}
    sphere { m*<-3.84117747700254,-7.736845412186243,-2.404610914631326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1697744580887686,0.6384286518924156,3.4031390434737325>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5 }
    cylinder { m*<3.663021647151304,0.6384286518924154,-0.8141431650168849>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5}
    cylinder { m*<-2.5994622370329497,6.044817556187686,-1.671095791362063>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5 }
    cylinder {  m*<-3.84117747700254,-7.736845412186243,-2.404610914631326>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5}

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
    sphere { m*<0.926130746962688,0.5891322197174367,0.41345754924973677>, 1 }        
    sphere {  m*<1.1697744580887686,0.6384286518924156,3.4031390434737325>, 1 }
    sphere {  m*<3.663021647151304,0.6384286518924154,-0.8141431650168849>, 1 }
    sphere {  m*<-2.5994622370329497,6.044817556187686,-1.671095791362063>, 1}
    sphere { m*<-3.84117747700254,-7.736845412186243,-2.404610914631326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1697744580887686,0.6384286518924156,3.4031390434737325>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5 }
    cylinder { m*<3.663021647151304,0.6384286518924154,-0.8141431650168849>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5}
    cylinder { m*<-2.5994622370329497,6.044817556187686,-1.671095791362063>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5 }
    cylinder {  m*<-3.84117747700254,-7.736845412186243,-2.404610914631326>, <0.926130746962688,0.5891322197174367,0.41345754924973677>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    