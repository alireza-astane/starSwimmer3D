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
    sphere { m*<-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 1 }        
    sphere {  m*<0.44830848020270825,0.2889014063980767,8.356653186976857>, 1 }
    sphere {  m*<3.3680170853395475,-0.0051139442509728905,-3.3746770380847466>, 1 }
    sphere {  m*<-2.107128493432416,2.182274642417158,-2.5468964388734303>, 1}
    sphere { m*<-1.8393412723945843,-2.705417299986739,-2.3573501537108594>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44830848020270825,0.2889014063980767,8.356653186976857>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5 }
    cylinder { m*<3.3680170853395475,-0.0051139442509728905,-3.3746770380847466>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5}
    cylinder { m*<-2.107128493432416,2.182274642417158,-2.5468964388734303>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5 }
    cylinder {  m*<-1.8393412723945843,-2.705417299986739,-2.3573501537108594>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5}

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
    sphere { m*<-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 1 }        
    sphere {  m*<0.44830848020270825,0.2889014063980767,8.356653186976857>, 1 }
    sphere {  m*<3.3680170853395475,-0.0051139442509728905,-3.3746770380847466>, 1 }
    sphere {  m*<-2.107128493432416,2.182274642417158,-2.5468964388734303>, 1}
    sphere { m*<-1.8393412723945843,-2.705417299986739,-2.3573501537108594>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44830848020270825,0.2889014063980767,8.356653186976857>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5 }
    cylinder { m*<3.3680170853395475,-0.0051139442509728905,-3.3746770380847466>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5}
    cylinder { m*<-2.107128493432416,2.182274642417158,-2.5468964388734303>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5 }
    cylinder {  m*<-1.8393412723945843,-2.705417299986739,-2.3573501537108594>, <-0.4749166376310239,-0.14632226642899257,-1.5911158242886392>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    