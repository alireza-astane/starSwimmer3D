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
    sphere { m*<-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 1 }        
    sphere {  m*<0.43550387925778433,-0.22550645822874357,9.142591414927887>, 1 }
    sphere {  m*<7.790855317257759,-0.3144267342230995,-5.436901875117446>, 1 }
    sphere {  m*<-6.164992664209821,5.184489921301035,-3.3660482507652536>, 1}
    sphere { m*<-2.241486959535258,-3.7051251164445302,-1.3322304287405995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.43550387925778433,-0.22550645822874357,9.142591414927887>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5 }
    cylinder { m*<7.790855317257759,-0.3144267342230995,-5.436901875117446>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5}
    cylinder { m*<-6.164992664209821,5.184489921301035,-3.3660482507652536>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5 }
    cylinder {  m*<-2.241486959535258,-3.7051251164445302,-1.3322304287405995>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5}

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
    sphere { m*<-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 1 }        
    sphere {  m*<0.43550387925778433,-0.22550645822874357,9.142591414927887>, 1 }
    sphere {  m*<7.790855317257759,-0.3144267342230995,-5.436901875117446>, 1 }
    sphere {  m*<-6.164992664209821,5.184489921301035,-3.3660482507652536>, 1}
    sphere { m*<-2.241486959535258,-3.7051251164445302,-1.3322304287405995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.43550387925778433,-0.22550645822874357,9.142591414927887>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5 }
    cylinder { m*<7.790855317257759,-0.3144267342230995,-5.436901875117446>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5}
    cylinder { m*<-6.164992664209821,5.184489921301035,-3.3660482507652536>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5 }
    cylinder {  m*<-2.241486959535258,-3.7051251164445302,-1.3322304287405995>, <-0.9965569523673243,-1.0453988079209497,-0.7203896580668945>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    