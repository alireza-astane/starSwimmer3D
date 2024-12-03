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
    sphere { m*<-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 1 }        
    sphere {  m*<0.12276532407989121,0.11358352369523805,2.7911533675076243>, 1 }
    sphere {  m*<2.616738613344462,0.08690742090128722,-1.4256109290641104>, 1 }
    sphere {  m*<-1.7395851405546923,2.313347389933515,-1.1703471690288962>, 1}
    sphere { m*<-1.6739695882525378,-2.956521106836175,-1.0979378729174867>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12276532407989121,0.11358352369523805,2.7911533675076243>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5 }
    cylinder { m*<2.616738613344462,0.08690742090128722,-1.4256109290641104>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5}
    cylinder { m*<-1.7395851405546923,2.313347389933515,-1.1703471690288962>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5 }
    cylinder {  m*<-1.6739695882525378,-2.956521106836175,-1.0979378729174867>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5}

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
    sphere { m*<-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 1 }        
    sphere {  m*<0.12276532407989121,0.11358352369523805,2.7911533675076243>, 1 }
    sphere {  m*<2.616738613344462,0.08690742090128722,-1.4256109290641104>, 1 }
    sphere {  m*<-1.7395851405546923,2.313347389933515,-1.1703471690288962>, 1}
    sphere { m*<-1.6739695882525378,-2.956521106836175,-1.0979378729174867>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12276532407989121,0.11358352369523805,2.7911533675076243>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5 }
    cylinder { m*<2.616738613344462,0.08690742090128722,-1.4256109290641104>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5}
    cylinder { m*<-1.7395851405546923,2.313347389933515,-1.1703471690288962>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5 }
    cylinder {  m*<-1.6739695882525378,-2.956521106836175,-1.0979378729174867>, <-0.11796978066180042,-0.015126554485087151,-0.1964014036129253>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    