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
    sphere { m*<-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 1 }        
    sphere {  m*<0.3001783416134166,0.16527386602701793,5.369552388133832>, 1 }
    sphere {  m*<2.536886048944028,0.0010495930176928348,-2.0399121071753923>, 1 }
    sphere {  m*<-1.8194377049551191,2.227489562049918,-1.7846483471401793>, 1}
    sphere { m*<-1.5516504839172873,-2.6602023803539794,-1.5951020619776066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3001783416134166,0.16527386602701793,5.369552388133832>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5 }
    cylinder { m*<2.536886048944028,0.0010495930176928348,-2.0399121071753923>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5}
    cylinder { m*<-1.8194377049551191,2.227489562049918,-1.7846483471401793>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5 }
    cylinder {  m*<-1.5516504839172873,-2.6602023803539794,-1.5951020619776066>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5}

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
    sphere { m*<-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 1 }        
    sphere {  m*<0.3001783416134166,0.16527386602701793,5.369552388133832>, 1 }
    sphere {  m*<2.536886048944028,0.0010495930176928348,-2.0399121071753923>, 1 }
    sphere {  m*<-1.8194377049551191,2.227489562049918,-1.7846483471401793>, 1}
    sphere { m*<-1.5516504839172873,-2.6602023803539794,-1.5951020619776066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3001783416134166,0.16527386602701793,5.369552388133832>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5 }
    cylinder { m*<2.536886048944028,0.0010495930176928348,-2.0399121071753923>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5}
    cylinder { m*<-1.8194377049551191,2.227489562049918,-1.7846483471401793>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5 }
    cylinder {  m*<-1.5516504839172873,-2.6602023803539794,-1.5951020619776066>, <-0.19782234506222907,-0.10098438236868136,-0.8107025817242111>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    