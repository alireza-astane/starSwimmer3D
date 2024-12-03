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
    sphere { m*<-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 1 }        
    sphere {  m*<0.21642590438802783,0.2840406113963447,8.55890101535323>, 1 }
    sphere {  m*<5.199047497236374,0.054212609889948526,-4.430161817029125>, 1 }
    sphere {  m*<-2.630166150306602,2.164199626156318,-2.2745680425244377>, 1}
    sphere { m*<-2.3623789292687705,-2.72349231624758,-2.0850217573618672>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21642590438802783,0.2840406113963447,8.55890101535323>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5 }
    cylinder { m*<5.199047497236374,0.054212609889948526,-4.430161817029125>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5}
    cylinder { m*<-2.630166150306602,2.164199626156318,-2.2745680425244377>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5 }
    cylinder {  m*<-2.3623789292687705,-2.72349231624758,-2.0850217573618672>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5}

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
    sphere { m*<-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 1 }        
    sphere {  m*<0.21642590438802783,0.2840406113963447,8.55890101535323>, 1 }
    sphere {  m*<5.199047497236374,0.054212609889948526,-4.430161817029125>, 1 }
    sphere {  m*<-2.630166150306602,2.164199626156318,-2.2745680425244377>, 1}
    sphere { m*<-2.3623789292687705,-2.72349231624758,-2.0850217573618672>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21642590438802783,0.2840406113963447,8.55890101535323>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5 }
    cylinder { m*<5.199047497236374,0.054212609889948526,-4.430161817029125>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5}
    cylinder { m*<-2.630166150306602,2.164199626156318,-2.2745680425244377>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5 }
    cylinder {  m*<-2.3623789292687705,-2.72349231624758,-2.0850217573618672>, <-0.9751592379792902,-0.16473201016340927,-1.3596794220229655>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    