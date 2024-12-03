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
    sphere { m*<0.24489597523073303,0.6708191329180978,0.013840725468963144>, 1 }        
    sphere {  m*<0.4856310799724247,0.7995292110984232,3.001395496589514>, 1 }
    sphere {  m*<2.9796043692369896,0.7728531083044721,-1.2153687999822207>, 1 }
    sphere {  m*<-1.3767193846621573,2.9992930773367,-0.9601050399470065>, 1}
    sphere { m*<-3.124330986333221,-5.698221391401304,-1.9382681143883156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4856310799724247,0.7995292110984232,3.001395496589514>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5 }
    cylinder { m*<2.9796043692369896,0.7728531083044721,-1.2153687999822207>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5}
    cylinder { m*<-1.3767193846621573,2.9992930773367,-0.9601050399470065>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5 }
    cylinder {  m*<-3.124330986333221,-5.698221391401304,-1.9382681143883156>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5}

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
    sphere { m*<0.24489597523073303,0.6708191329180978,0.013840725468963144>, 1 }        
    sphere {  m*<0.4856310799724247,0.7995292110984232,3.001395496589514>, 1 }
    sphere {  m*<2.9796043692369896,0.7728531083044721,-1.2153687999822207>, 1 }
    sphere {  m*<-1.3767193846621573,2.9992930773367,-0.9601050399470065>, 1}
    sphere { m*<-3.124330986333221,-5.698221391401304,-1.9382681143883156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4856310799724247,0.7995292110984232,3.001395496589514>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5 }
    cylinder { m*<2.9796043692369896,0.7728531083044721,-1.2153687999822207>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5}
    cylinder { m*<-1.3767193846621573,2.9992930773367,-0.9601050399470065>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5 }
    cylinder {  m*<-3.124330986333221,-5.698221391401304,-1.9382681143883156>, <0.24489597523073303,0.6708191329180978,0.013840725468963144>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    