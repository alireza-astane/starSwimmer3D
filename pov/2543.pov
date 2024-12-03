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
    sphere { m*<0.8553962130748947,0.6976349456913922,0.3716348638655092>, 1 }        
    sphere {  m*<1.098744715320413,0.7577824740199599,3.361141371512365>, 1 }
    sphere {  m*<3.591991904382949,0.7577824740199597,-0.8561408369782526>, 1 }
    sphere {  m*<-2.371297434546957,5.6260758012840375,-1.5361875607675095>, 1}
    sphere { m*<-3.8668981589562192,-7.662897873407688,-2.419820053852323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.098744715320413,0.7577824740199599,3.361141371512365>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5 }
    cylinder { m*<3.591991904382949,0.7577824740199597,-0.8561408369782526>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5}
    cylinder { m*<-2.371297434546957,5.6260758012840375,-1.5361875607675095>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5 }
    cylinder {  m*<-3.8668981589562192,-7.662897873407688,-2.419820053852323>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5}

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
    sphere { m*<0.8553962130748947,0.6976349456913922,0.3716348638655092>, 1 }        
    sphere {  m*<1.098744715320413,0.7577824740199599,3.361141371512365>, 1 }
    sphere {  m*<3.591991904382949,0.7577824740199597,-0.8561408369782526>, 1 }
    sphere {  m*<-2.371297434546957,5.6260758012840375,-1.5361875607675095>, 1}
    sphere { m*<-3.8668981589562192,-7.662897873407688,-2.419820053852323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.098744715320413,0.7577824740199599,3.361141371512365>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5 }
    cylinder { m*<3.591991904382949,0.7577824740199597,-0.8561408369782526>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5}
    cylinder { m*<-2.371297434546957,5.6260758012840375,-1.5361875607675095>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5 }
    cylinder {  m*<-3.8668981589562192,-7.662897873407688,-2.419820053852323>, <0.8553962130748947,0.6976349456913922,0.3716348638655092>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    