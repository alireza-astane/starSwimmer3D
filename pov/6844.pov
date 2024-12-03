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
    sphere { m*<-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 1 }        
    sphere {  m*<0.48349910068097546,-0.2653104119499341,9.167049286376535>, 1 }
    sphere {  m*<7.838850538680952,-0.35423068794429,-5.412444003668796>, 1 }
    sphere {  m*<-6.378770944660095,5.391474034227649,-3.475168905238545>, 1}
    sphere { m*<-2.1817913487839644,-3.7720628591580367,-1.3017036435681917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.48349910068097546,-0.2653104119499341,9.167049286376535>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5 }
    cylinder { m*<7.838850538680952,-0.35423068794429,-5.412444003668796>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5}
    cylinder { m*<-6.378770944660095,5.391474034227649,-3.475168905238545>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5 }
    cylinder {  m*<-2.1817913487839644,-3.7720628591580367,-1.3017036435681917>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5}

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
    sphere { m*<-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 1 }        
    sphere {  m*<0.48349910068097546,-0.2653104119499341,9.167049286376535>, 1 }
    sphere {  m*<7.838850538680952,-0.35423068794429,-5.412444003668796>, 1 }
    sphere {  m*<-6.378770944660095,5.391474034227649,-3.475168905238545>, 1}
    sphere { m*<-2.1817913487839644,-3.7720628591580367,-1.3017036435681917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.48349910068097546,-0.2653104119499341,9.167049286376535>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5 }
    cylinder { m*<7.838850538680952,-0.35423068794429,-5.412444003668796>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5}
    cylinder { m*<-6.378770944660095,5.391474034227649,-3.475168905238545>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5 }
    cylinder {  m*<-2.1817913487839644,-3.7720628591580367,-1.3017036435681917>, <-0.9459314882015052,-1.1070552302274441,-0.69447123270104>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    