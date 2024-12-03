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
    sphere { m*<-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 1 }        
    sphere {  m*<0.5213610533463822,0.2905191741255736,8.295171228007368>, 1 }
    sphere {  m*<2.6229558777980753,-0.030704514945560896,-2.9826866458227324>, 1 }
    sphere {  m*<-1.9239591339156552,2.188973550479282,-2.6327260131727535>, 1}
    sphere { m*<-1.6561719128778234,-2.6987183919246154,-2.4431797280101826>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5213610533463822,0.2905191741255736,8.295171228007368>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5 }
    cylinder { m*<2.6229558777980753,-0.030704514945560896,-2.9826866458227324>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5}
    cylinder { m*<-1.9239591339156552,2.188973550479282,-2.6327260131727535>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5 }
    cylinder {  m*<-1.6561719128778234,-2.6987183919246154,-2.4431797280101826>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5}

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
    sphere { m*<-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 1 }        
    sphere {  m*<0.5213610533463822,0.2905191741255736,8.295171228007368>, 1 }
    sphere {  m*<2.6229558777980753,-0.030704514945560896,-2.9826866458227324>, 1 }
    sphere {  m*<-1.9239591339156552,2.188973550479282,-2.6327260131727535>, 1}
    sphere { m*<-1.6561719128778234,-2.6987183919246154,-2.4431797280101826>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5213610533463822,0.2905191741255736,8.295171228007368>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5 }
    cylinder { m*<2.6229558777980753,-0.030704514945560896,-2.9826866458227324>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5}
    cylinder { m*<-1.9239591339156552,2.188973550479282,-2.6327260131727535>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5 }
    cylinder {  m*<-1.6561719128778234,-2.6987183919246154,-2.4431797280101826>, <-0.3006163643553033,-0.13951911501875694,-1.661707237488195>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    