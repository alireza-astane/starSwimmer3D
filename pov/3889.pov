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
    sphere { m*<-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 1 }        
    sphere {  m*<0.14716227457744874,0.15970246081683448,2.8052888066677504>, 1 }
    sphere {  m*<2.6411355638420204,0.13302635802288365,-1.4114754899039856>, 1 }
    sphere {  m*<-1.715188190057134,2.3594663270551117,-1.156211729868771>, 1}
    sphere { m*<-1.7948835246587551,-3.185091568165001,-1.167994644637893>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14716227457744874,0.15970246081683448,2.8052888066677504>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5 }
    cylinder { m*<2.6411355638420204,0.13302635802288365,-1.4114754899039856>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5}
    cylinder { m*<-1.715188190057134,2.3594663270551117,-1.156211729868771>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5 }
    cylinder {  m*<-1.7948835246587551,-3.185091568165001,-1.167994644637893>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5}

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
    sphere { m*<-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 1 }        
    sphere {  m*<0.14716227457744874,0.15970246081683448,2.8052888066677504>, 1 }
    sphere {  m*<2.6411355638420204,0.13302635802288365,-1.4114754899039856>, 1 }
    sphere {  m*<-1.715188190057134,2.3594663270551117,-1.156211729868771>, 1}
    sphere { m*<-1.7948835246587551,-3.185091568165001,-1.167994644637893>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14716227457744874,0.15970246081683448,2.8052888066677504>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5 }
    cylinder { m*<2.6411355638420204,0.13302635802288365,-1.4114754899039856>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5}
    cylinder { m*<-1.715188190057134,2.3594663270551117,-1.156211729868771>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5 }
    cylinder {  m*<-1.7948835246587551,-3.185091568165001,-1.167994644637893>, <-0.09357283016424278,0.030992382636509275,-0.1822659644527997>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    