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
    sphere { m*<0.9339902667260294,0.5768690755168001,0.41810459822231527>, 1 }        
    sphere {  m*<1.1776627317446906,0.6249778400315824,3.407803142003794>, 1 }
    sphere {  m*<3.670909920807227,0.6249778400315822,-0.8094790664868245>, 1 }
    sphere {  m*<-2.6245198983275975,6.0913744842305,-1.68591179373475>, 1}
    sphere { m*<-3.838232929752428,-7.745306961674493,-2.4028697467768776>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1776627317446906,0.6249778400315824,3.407803142003794>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5 }
    cylinder { m*<3.670909920807227,0.6249778400315822,-0.8094790664868245>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5}
    cylinder { m*<-2.6245198983275975,6.0913744842305,-1.68591179373475>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5 }
    cylinder {  m*<-3.838232929752428,-7.745306961674493,-2.4028697467768776>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5}

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
    sphere { m*<0.9339902667260294,0.5768690755168001,0.41810459822231527>, 1 }        
    sphere {  m*<1.1776627317446906,0.6249778400315824,3.407803142003794>, 1 }
    sphere {  m*<3.670909920807227,0.6249778400315822,-0.8094790664868245>, 1 }
    sphere {  m*<-2.6245198983275975,6.0913744842305,-1.68591179373475>, 1}
    sphere { m*<-3.838232929752428,-7.745306961674493,-2.4028697467768776>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1776627317446906,0.6249778400315824,3.407803142003794>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5 }
    cylinder { m*<3.670909920807227,0.6249778400315822,-0.8094790664868245>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5}
    cylinder { m*<-2.6245198983275975,6.0913744842305,-1.68591179373475>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5 }
    cylinder {  m*<-3.838232929752428,-7.745306961674493,-2.4028697467768776>, <0.9339902667260294,0.5768690755168001,0.41810459822231527>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    