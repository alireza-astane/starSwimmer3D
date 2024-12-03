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
    sphere { m*<-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 1 }        
    sphere {  m*<0.14879345851071607,0.16278598017961943,2.8062339043450186>, 1 }
    sphere {  m*<2.6427667477752874,0.1361098773856686,-1.410530392226717>, 1 }
    sphere {  m*<-1.7135570061238665,2.3625498464178967,-1.1552666321915024>, 1}
    sphere { m*<-1.8027291376974741,-3.199922574788114,-1.1725403433223982>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14879345851071607,0.16278598017961943,2.8062339043450186>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5 }
    cylinder { m*<2.6427667477752874,0.1361098773856686,-1.410530392226717>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5}
    cylinder { m*<-1.7135570061238665,2.3625498464178967,-1.1552666321915024>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5 }
    cylinder {  m*<-1.8027291376974741,-3.199922574788114,-1.1725403433223982>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5}

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
    sphere { m*<-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 1 }        
    sphere {  m*<0.14879345851071607,0.16278598017961943,2.8062339043450186>, 1 }
    sphere {  m*<2.6427667477752874,0.1361098773856686,-1.410530392226717>, 1 }
    sphere {  m*<-1.7135570061238665,2.3625498464178967,-1.1552666321915024>, 1}
    sphere { m*<-1.8027291376974741,-3.199922574788114,-1.1725403433223982>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14879345851071607,0.16278598017961943,2.8062339043450186>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5 }
    cylinder { m*<2.6427667477752874,0.1361098773856686,-1.410530392226717>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5}
    cylinder { m*<-1.7135570061238665,2.3625498464178967,-1.1552666321915024>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5 }
    cylinder {  m*<-1.8027291376974741,-3.199922574788114,-1.1725403433223982>, <-0.09194164623097545,0.03407590199929422,-0.18132086677553116>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    