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
    sphere { m*<-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 1 }        
    sphere {  m*<0.10944256105641387,0.06329614552124091,3.0024959015532535>, 1 }
    sphere {  m*<2.5815089557894852,0.02490742561288023,-1.4861358791904506>, 1 }
    sphere {  m*<-1.7748147981096618,2.251347394645105,-1.2308721191552372>, 1}
    sphere { m*<-1.50702757707183,-2.6363445477587923,-1.0413258339926645>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10944256105641387,0.06329614552124091,3.0024959015532535>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5 }
    cylinder { m*<2.5815089557894852,0.02490742561288023,-1.4861358791904506>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5}
    cylinder { m*<-1.7748147981096618,2.251347394645105,-1.2308721191552372>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5 }
    cylinder {  m*<-1.50702757707183,-2.6363445477587923,-1.0413258339926645>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5}

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
    sphere { m*<-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 1 }        
    sphere {  m*<0.10944256105641387,0.06329614552124091,3.0024959015532535>, 1 }
    sphere {  m*<2.5815089557894852,0.02490742561288023,-1.4861358791904506>, 1 }
    sphere {  m*<-1.7748147981096618,2.251347394645105,-1.2308721191552372>, 1}
    sphere { m*<-1.50702757707183,-2.6363445477587923,-1.0413258339926645>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10944256105641387,0.06329614552124091,3.0024959015532535>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5 }
    cylinder { m*<2.5815089557894852,0.02490742561288023,-1.4861358791904506>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5}
    cylinder { m*<-1.7748147981096618,2.251347394645105,-1.2308721191552372>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5 }
    cylinder {  m*<-1.50702757707183,-2.6363445477587923,-1.0413258339926645>, <-0.15319943821677157,-0.07712654977349388,-0.25692635373926676>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    