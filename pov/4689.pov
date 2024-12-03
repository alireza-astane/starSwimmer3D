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
    sphere { m*<-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 1 }        
    sphere {  m*<0.4017590134212028,0.2195844171367906,6.630182079173756>, 1 }
    sphere {  m*<2.508148784051645,-0.014314911527483924,-2.3965453989020604>, 1 }
    sphere {  m*<-1.848174969847502,2.212125057504741,-2.141281638866847>, 1}
    sphere { m*<-1.5803877488096703,-2.6755668848991565,-1.9517353537042745>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4017590134212028,0.2195844171367906,6.630182079173756>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5 }
    cylinder { m*<2.508148784051645,-0.014314911527483924,-2.3965453989020604>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5}
    cylinder { m*<-1.848174969847502,2.212125057504741,-2.141281638866847>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5 }
    cylinder {  m*<-1.5803877488096703,-2.6755668848991565,-1.9517353537042745>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5}

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
    sphere { m*<-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 1 }        
    sphere {  m*<0.4017590134212028,0.2195844171367906,6.630182079173756>, 1 }
    sphere {  m*<2.508148784051645,-0.014314911527483924,-2.3965453989020604>, 1 }
    sphere {  m*<-1.848174969847502,2.212125057504741,-2.141281638866847>, 1}
    sphere { m*<-1.5803877488096703,-2.6755668848991565,-1.9517353537042745>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4017590134212028,0.2195844171367906,6.630182079173756>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5 }
    cylinder { m*<2.508148784051645,-0.014314911527483924,-2.3965453989020604>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5}
    cylinder { m*<-1.848174969847502,2.212125057504741,-2.141281638866847>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5 }
    cylinder {  m*<-1.5803877488096703,-2.6755668848991565,-1.9517353537042745>, <-0.22655960995461205,-0.11634888691385815,-1.16733587345088>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    