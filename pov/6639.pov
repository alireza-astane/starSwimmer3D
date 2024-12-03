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
    sphere { m*<-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 1 }        
    sphere {  m*<0.31448985978876753,-0.1252923278912242,9.080924810611796>, 1 }
    sphere {  m*<7.669841297788735,-0.21421260388558105,-5.498568479433542>, 1 }
    sphere {  m*<-5.6118620102101,4.641608478791721,-3.083664302657678>, 1}
    sphere { m*<-2.3945897441275723,-3.5290973709821314,-1.4105498050959897>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.31448985978876753,-0.1252923278912242,9.080924810611796>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5 }
    cylinder { m*<7.669841297788735,-0.21421260388558105,-5.498568479433542>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5}
    cylinder { m*<-5.6118620102101,4.641608478791721,-3.083664302657678>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5 }
    cylinder {  m*<-2.3945897441275723,-3.5290973709821314,-1.4105498050959897>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5}

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
    sphere { m*<-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 1 }        
    sphere {  m*<0.31448985978876753,-0.1252923278912242,9.080924810611796>, 1 }
    sphere {  m*<7.669841297788735,-0.21421260388558105,-5.498568479433542>, 1 }
    sphere {  m*<-5.6118620102101,4.641608478791721,-3.083664302657678>, 1}
    sphere { m*<-2.3945897441275723,-3.5290973709821314,-1.4105498050959897>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.31448985978876753,-0.1252923278912242,9.080924810611796>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5 }
    cylinder { m*<7.669841297788735,-0.21421260388558105,-5.498568479433542>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5}
    cylinder { m*<-5.6118620102101,4.641608478791721,-3.083664302657678>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5 }
    cylinder {  m*<-2.3945897441275723,-3.5290973709821314,-1.4105498050959897>, <-1.1245012784830721,-0.884280620763981,-0.7859249342091789>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    