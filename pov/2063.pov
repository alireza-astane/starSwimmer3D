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
    sphere { m*<1.2301327555861157,0.08640904281734436,0.5932041138646972>, 1 }        
    sphere {  m*<1.4743704414610335,0.09274531533883416,3.5832385848838646>, 1 }
    sphere {  m*<3.967617630523571,0.09274531533883416,-0.6340436236067526>, 1 }
    sphere {  m*<-3.541557946156518,7.865334984908124,-2.228137669257472>, 1}
    sphere { m*<-3.713472255053983,-8.099614634639057,-2.329096520297635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4743704414610335,0.09274531533883416,3.5832385848838646>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5 }
    cylinder { m*<3.967617630523571,0.09274531533883416,-0.6340436236067526>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5}
    cylinder { m*<-3.541557946156518,7.865334984908124,-2.228137669257472>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5 }
    cylinder {  m*<-3.713472255053983,-8.099614634639057,-2.329096520297635>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5}

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
    sphere { m*<1.2301327555861157,0.08640904281734436,0.5932041138646972>, 1 }        
    sphere {  m*<1.4743704414610335,0.09274531533883416,3.5832385848838646>, 1 }
    sphere {  m*<3.967617630523571,0.09274531533883416,-0.6340436236067526>, 1 }
    sphere {  m*<-3.541557946156518,7.865334984908124,-2.228137669257472>, 1}
    sphere { m*<-3.713472255053983,-8.099614634639057,-2.329096520297635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4743704414610335,0.09274531533883416,3.5832385848838646>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5 }
    cylinder { m*<3.967617630523571,0.09274531533883416,-0.6340436236067526>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5}
    cylinder { m*<-3.541557946156518,7.865334984908124,-2.228137669257472>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5 }
    cylinder {  m*<-3.713472255053983,-8.099614634639057,-2.329096520297635>, <1.2301327555861157,0.08640904281734436,0.5932041138646972>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    